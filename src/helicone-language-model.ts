import {
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
} from '@ai-sdk/provider';
import { asSchema } from '@ai-sdk/provider-utils';
import { HeliconeSettings, HeliconeExtraBody } from './types';
import { convertToHeliconePrompt } from './convert-to-helicone-prompt';
import {
  HeliconeErrorData,
  mapHeliconeFinishReason,
  createHeliconeError,
} from './helicone-error';

type HeliconeLanguageModelConfig = {
  modelId: string;
  settings: HeliconeSettings;
  extraBody?: HeliconeExtraBody;
};

export class HeliconeLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = 'v2' as const;
  readonly supportedUrls: Record<string, RegExp[]> = {};
  readonly provider: string = 'helicone';
  readonly modelId: string;
  private readonly settings: HeliconeSettings;
  private readonly extraBody?: HeliconeExtraBody;

  constructor(config: HeliconeLanguageModelConfig) {
    this.modelId = config.modelId;
    this.settings = config.settings;
    this.extraBody = config.extraBody;
  }

  private get baseURL(): string {
    return this.settings.baseURL ?? 'https://ai-gateway.helicone.ai';
  }

  private get headers(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...(this.settings.headers || {})
    };

    if (this.settings.apiKey) {
      headers['Authorization'] = `Bearer ${this.settings.apiKey}`;
    }

    // Convert Helicone metadata from extraBody to headers
    if (this.extraBody?.helicone) {
      const helicone = this.extraBody.helicone;

      if (helicone.sessionId) {
        headers['Helicone-Session-Id'] = helicone.sessionId;
      }

      if (helicone.userId) {
        headers['Helicone-User-Id'] = helicone.userId;
      }

      if (helicone.properties) {
        Object.entries(helicone.properties).forEach(([key, value]) => {
          headers[`Helicone-Property-${key}`] = String(value);
        });
      }

      if (helicone.tags && helicone.tags.length > 0) {
        helicone.tags.forEach((tag) => {
          headers[`Helicone-Property-Tag-${tag}`] = 'true';
        });
      }

      if (helicone.cache !== undefined) {
        headers['Helicone-Cache-Enabled'] = String(helicone.cache);
      }
    }

    return headers;
  }

  /**
   * Normalize tool schemas from the AI SDK into JSON Schema for Helicone.
   * Supports both JSON Schema objects and zod/standard schemas.
   */
  private normalizeToolParameters(tool: any): any {
    const schemaSource = tool?.parameters ?? tool?.inputSchema;

    // Fallback when no schema is provided
    if (!schemaSource) {
      return { type: 'object', properties: {} };
    }

    // Schemas created with jsonSchema/zodSchema already expose .jsonSchema
    if (schemaSource.jsonSchema) {
      return schemaSource.jsonSchema;
    }

    const isZodLike =
      typeof schemaSource === 'object' &&
      schemaSource !== null &&
      !Array.isArray(schemaSource) &&
      ('_def' in schemaSource || typeof schemaSource.safeParse === 'function');

    // If it already looks like JSON Schema, pass it through untouched
    const looksLikeJsonSchema =
      typeof schemaSource === 'object' &&
      schemaSource !== null &&
      !Array.isArray(schemaSource) &&
      !isZodLike &&
      (schemaSource.type ||
        schemaSource.properties ||
        schemaSource.oneOf ||
        schemaSource.anyOf ||
        schemaSource.allOf ||
        schemaSource.$ref);

    if (looksLikeJsonSchema) {
      return schemaSource;
    }

    // Otherwise try converting (e.g., zod schema)
    try {
      return asSchema(schemaSource).jsonSchema;
    } catch (error) {
      // Last resort: coerce to a minimal object schema
      if (typeof schemaSource === 'object') {
        return {
          type: 'object',
          ...schemaSource
        };
      }
      throw error;
    }
  }

  private buildRequestBody(options: LanguageModelV2CallOptions): any {
    // Extract helicone metadata and other extraBody fields
    const { helicone, prompt_id, inputs, environment, ...otherExtraBody } = this.extraBody || {};

    const body: any = {
      model: this.modelId,
      stream: false,
      ...(options.temperature != null && { temperature: options.temperature }),
      ...(options.maxOutputTokens != null && { max_tokens: options.maxOutputTokens }),
      ...(options.topP != null && { top_p: options.topP }),
      ...(options.topK != null && { top_k: options.topK }),
      ...(options.frequencyPenalty != null && { frequency_penalty: options.frequencyPenalty }),
      ...(options.presencePenalty != null && { presence_penalty: options.presencePenalty }),
      ...(options.stopSequences != null && { stop: options.stopSequences }),
      ...(options.seed != null && { seed: options.seed }),
      ...otherExtraBody
    };

    // Handle Helicone prompt integration
    if (prompt_id) {
      // Use Helicone prompt integration - replace messages with prompt_id and inputs
      body.prompt_id = prompt_id;
      if (environment) {
        body.environment = environment;
      }
      if (inputs) {
        body.inputs = inputs;
      }
      // Don't include messages when using promptId
    } else if (options.prompt && options.prompt.length > 0) {
      // Use regular messages format only if we have a prompt
      const prompt = convertToHeliconePrompt(options.prompt);
      body.messages = prompt;
    }

    if (options.toolChoice) {
      if (options.toolChoice.type === 'auto') {
        body.tool_choice = 'auto';
      } else if (options.toolChoice.type === 'required') {
        body.tool_choice = 'required';
      } else if (options.toolChoice.type === 'none') {
        body.tool_choice = 'none';
      } else if (options.toolChoice.type === 'tool') {
        body.tool_choice = {
          type: 'function',
          function: { name: options.toolChoice.toolName },
        };
      }
    }

    if (options.tools && options.tools.length > 0) {
      body.tools = options.tools.map((tool: any) => {
        const parameters = this.normalizeToolParameters(tool);

        // Validate and ensure type is always "object"
        if (!parameters.type || parameters.type === 'None' || typeof parameters.type !== 'string') {
          parameters.type = 'object';
        }

        // Ensure required fields exist
        if (!parameters.properties && !parameters.additionalProperties) {
          parameters.properties = {};
        }

        // Validate the schema before sending
        const toolName = tool.name || tool.toolName;
        if (parameters.type !== 'object') {
          throw createHeliconeError({
            message: `Invalid schema for function '${toolName}': schema must be a JSON Schema of 'type: "object"', got 'type: "${parameters.type}"'.`
          });
        }

        return {
          type: 'function',
          function: {
            name: toolName,
            description: tool.description || '',
            parameters
          }
        };
      });
    }

    return body;
  }

  async doGenerate(options: LanguageModelV2CallOptions) {
    const body = this.buildRequestBody(options);

    try {
      const response = await fetch(`${this.baseURL}/v1/chat/completions`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => '');
        let errorData: HeliconeErrorData = {};

        try {
          errorData = JSON.parse(errorText);
        } catch {
          // If not JSON, put the raw text in the error message
          errorData = {
            error: {
              message: errorText || `HTTP ${response.status}: ${response.statusText}`,
              type: 'http_error',
              code: response.status.toString(),
            },
          };
        }


        throw createHeliconeError({
          data: errorData,
          response,
        });
      }

      const data: any = await response.json();

      const choice = data.choices[0];
      const message = choice.message;

      const content: Array<any> = [];

      if (message.content) {
        content.push({
          type: 'text',
          text: message.content
        });
      }

      if (message.tool_calls) {
        for (const toolCall of message.tool_calls) {
          let parsedInput: any;
          try {
            parsedInput = JSON.parse(toolCall.function.arguments || '{}');
          } catch (error) {
            // If parsing fails, use empty object as fallback
            parsedInput = {};
          }
          content.push({
            type: 'tool-call',
            toolCallId: toolCall.id,
            toolName: toolCall.function.name,
            input: parsedInput
          });
        }
      }

      return {
        content,
        finishReason: mapHeliconeFinishReason(choice.finish_reason),
        usage: {
          inputTokens: data.usage?.prompt_tokens ?? 0,
          outputTokens: data.usage?.completion_tokens ?? 0,
          totalTokens: data.usage?.total_tokens ?? (data.usage?.prompt_tokens ?? 0) + (data.usage?.completion_tokens ?? 0)
        },
        warnings: []
      };
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }

      throw createHeliconeError({
        message: `Failed to generate response: ${error}`,
        cause: error,
      });
    }
  }

  async doStream(options: LanguageModelV2CallOptions) {
    const body = { ...this.buildRequestBody(options), stream: true };

    try {
      const response = await fetch(`${this.baseURL}/v1/chat/completions`, {
        method: 'POST',
        headers: this.headers,
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({})) as HeliconeErrorData;
        throw createHeliconeError({
          data: errorData,
          response,
        });
      }

      const usage = { inputTokens: 0, outputTokens: 0, totalTokens: 0 };
      let actualFinishReason: LanguageModelV2FinishReason = 'stop';

      // Track tool calls to send completion events
      const toolCalls: Map<string, {
        id: string;
        name: string;
        arguments: string;
        completed: boolean;
      }> = new Map();

      // Map tool call index to ID for streaming chunks
      const indexToId: Map<number, string> = new Map();

      // Track text parts for start/end events
      const textStarted = new Set<string>();
      const textEnded = new Set<string>();

      const textDecoder = new TextDecoder();
      const reader = response.body!.getReader();

      const processedStream = new ReadableStream<LanguageModelV2StreamPart>({
        async start(controller) {
          try {
            while (true) {
              const { done, value } = await reader.read();
              if (done) {
                // Complete any outstanding tool calls before finishing
                for (const [id, toolCall] of toolCalls) {
                  if (!toolCall.completed) {
                    controller.enqueue({
                      type: 'tool-input-end',
                      id: toolCall.id
                    } as LanguageModelV2StreamPart);

                    controller.enqueue({
                      type: 'tool-call',
                      toolCallId: toolCall.id,
                      toolName: toolCall.name,
                      input: toolCall.arguments
                    } as LanguageModelV2StreamPart);

                    toolCall.completed = true;
                  }
                }

                // Complete any outstanding text parts before finishing
                for (const id of textStarted) {
                  if (!textEnded.has(id)) {
                    controller.enqueue({
                      type: 'text-end',
                      id
                    } as LanguageModelV2StreamPart);
                    textEnded.add(id);
                  }
                }

                controller.enqueue({
                  type: 'finish',
                  usage: usage as LanguageModelV2Usage,
                  finishReason: actualFinishReason,
                });
                controller.close();
                break;
              }

              const chunk = textDecoder.decode(value);

              // Process each line in the chunk
              for (const line of chunk.split('\n')) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data: ')) continue;

                const data = trimmed.slice(6);
                if (data === '[DONE]') {
                  continue;
                }

                // Transform OpenAI streaming format to AI SDK events
                try {
                  const parsed = JSON.parse(data);

                  // Update usage data when available (even if no choices)
                  if (parsed.usage) {
                    usage.inputTokens = parsed.usage.prompt_tokens || 0;
                    usage.outputTokens = parsed.usage.completion_tokens || 0;
                    usage.totalTokens = parsed.usage.total_tokens || 0;
                  }

                  const choice = parsed.choices?.[0];
                  if (!choice) continue;

                  // Capture finish reason and complete tool calls
                  if (choice.finish_reason) {
                    actualFinishReason = mapHeliconeFinishReason(choice.finish_reason);

                    // Complete any outstanding tool calls
                    for (const [id, toolCall] of toolCalls) {
                      if (!toolCall.completed) {
                        controller.enqueue({
                          type: 'tool-input-end',
                          id: toolCall.id
                        } as LanguageModelV2StreamPart);

                        controller.enqueue({
                          type: 'tool-call',
                          toolCallId: toolCall.id,
                          toolName: toolCall.name,
                          input: toolCall.arguments
                        } as LanguageModelV2StreamPart);

                        toolCall.completed = true;
                      }
                    }

                    // Complete any outstanding text parts
                    for (const id of textStarted) {
                      if (!textEnded.has(id)) {
                        controller.enqueue({
                          type: 'text-end',
                          id
                        } as LanguageModelV2StreamPart);
                        textEnded.add(id);
                      }
                    }
                  }

                  const delta = choice.delta;
                  if (!delta) continue;

                  // Transform text content to text-delta events
                  if (delta.content) {
                    const id = 'text-0';
                    // Emit text-start before first text-delta
                    if (!textStarted.has(id)) {
                      controller.enqueue({
                        type: 'text-start',
                        id
                      } as LanguageModelV2StreamPart);
                      textStarted.add(id);
                    }
                    controller.enqueue({
                      type: 'text-delta',
                      delta: delta.content,
                      id
                    } as LanguageModelV2StreamPart);
                  }

                  // Transform tool calls to tool-input events
                  if (delta.tool_calls) {
                    for (const toolCall of delta.tool_calls) {
                      let toolId: string;

                      if (toolCall.id) {
                        // First chunk with ID, store the mapping
                        toolId = toolCall.id;
                        if (toolCall.index !== undefined) {
                          indexToId.set(toolCall.index, toolId);
                        }
                      } else if (toolCall.index !== undefined && indexToId.has(toolCall.index)) {
                        // Subsequent chunks, use mapped ID
                        toolId = indexToId.get(toolCall.index)!;
                      } else {
                        continue;
                      }

                      // Get or create tool call tracking
                      let trackedCall = toolCalls.get(toolId);
                      if (!trackedCall) {
                        trackedCall = {
                          id: toolId,
                          name: '',
                          arguments: '{}',
                          completed: false
                        };
                        toolCalls.set(toolId, trackedCall);
                      }

                      // Send tool-input-start when we get the tool name
                      if (toolCall.function?.name && !trackedCall.name) {
                        trackedCall.name = toolCall.function.name;
                        controller.enqueue({
                          type: 'tool-input-start',
                          id: toolId,
                          toolName: toolCall.function.name
                        } as LanguageModelV2StreamPart);
                      }

                      // Send tool-input-delta when we get arguments
                      if (toolCall.function?.arguments) {
                        // Smart argument accumulation
                        if (trackedCall.arguments === '{}') {
                          trackedCall.arguments = toolCall.function.arguments;  // Replace default
                        } else {
                          trackedCall.arguments += toolCall.function.arguments;  // Accumulate
                        }

                        controller.enqueue({
                          type: 'tool-input-delta',
                          id: toolId,
                          delta: toolCall.function.arguments
                        } as LanguageModelV2StreamPart);
                      }
                    }
                  }

                } catch (parseError) {
                  // Skip malformed JSON chunks
                }
              }
            }
          } catch (error) {
            controller.error(error);
          }
        }
      });

      return {
        stream: processedStream,
        rawCall: { rawPrompt: body.messages, rawSettings: body }
      };

    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw error;
      }
      throw createHeliconeError({
        message: `Failed to stream response: ${error}`,
        cause: error
      });
    }
  }
}
