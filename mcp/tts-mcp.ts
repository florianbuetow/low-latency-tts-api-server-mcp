/**
 * MCP server that relays requests to the Kokoro FastAPI speech server.
 *
 * This layer is intentionally thin: it loads host/port from config.yaml,
 * checks /health before each request, and returns JSON responses unchanged.
 */

import { readFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const PROJECT_ROOT = resolve(__dirname, "..");
const CONFIG_PATH = process.env.KOKORO_TTS_CONFIG_PATH === undefined
  ? resolve(PROJECT_ROOT, "config.yaml")
  : resolve(process.env.KOKORO_TTS_CONFIG_PATH);
const HEALTH_TIMEOUT_MS = 3_000;
const REQUEST_TIMEOUT_MS = 30_000;

type ToolResult = {
  content: Array<{ type: "text"; text: string }>;
  isError?: boolean;
};

function requireConfigMatch(content: string, key: string, pattern: RegExp): string {
  const match = content.match(pattern);
  if (!match || !match[1]) {
    throw new Error(`Missing required key '${key}' in ${CONFIG_PATH}`);
  }
  return match[1].trim();
}

function loadServerUrl(): string {
  let content: string;
  try {
    content = readFileSync(CONFIG_PATH, "utf-8");
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    throw new Error(`Cannot read config file ${CONFIG_PATH}: ${detail}`);
  }

  const host = requireConfigMatch(content, "host", /^host:\s*(.+)$/m);
  const port = requireConfigMatch(content, "port", /^port:\s*(\d+)$/m);
  const connectHost = host === "0.0.0.0" ? "127.0.0.1" : host;
  return `http://${connectHost}:${port}`;
}

async function healthCheck(): Promise<ToolResult | null> {
  const baseUrl = loadServerUrl();
  const url = `${baseUrl}/health`;
  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!response.ok) {
      const error = {
        error: "health_check_failed",
        url,
        message: `Kokoro speech server health check failed: HTTP ${response.status}`,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
        isError: true,
      };
    }
    const body = await response.json() as { status?: string };
    if (body?.status !== "ok") {
      const error = {
        error: "health_check_failed",
        url,
        message: "Kokoro speech server reported unhealthy status",
        details: body,
      };
      return {
        content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
        isError: true,
      };
    }
    return null;
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    const error = {
      error: "health_check_unreachable",
      url,
      message: `Kokoro speech server is not reachable at ${baseUrl}`,
      details: detail,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }
}

async function request(
  method: "GET" | "POST",
  path: string,
  body?: Record<string, unknown>,
): Promise<ToolResult> {
  const healthResult = await healthCheck();
  if (healthResult !== null) {
    return healthResult;
  }

  const baseUrl = loadServerUrl();
  const url = `${baseUrl}${path}`;

  let response: Response;
  try {
    const options: RequestInit = {
      method,
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    };
    if (method === "POST" && body !== undefined) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }
    response = await fetch(url, options);
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    const error = {
      error: "connection_failed",
      url,
      message: `Kokoro speech server is not reachable at ${baseUrl}`,
      details: detail,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }

  const responseText = await response.text();
  let responseBody: unknown;
  try {
    responseBody = JSON.parse(responseText);
  } catch {
    const error = {
      error: "invalid_json",
      status_code: response.status,
      url,
      raw_body: responseText.slice(0, 500),
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }

  if (!response.ok) {
    const error = {
      error: "http_error",
      status_code: response.status,
      url,
      response: responseBody,
    };
    return {
      content: [{ type: "text", text: JSON.stringify(error, null, 2) }],
      isError: true,
    };
  }

  return {
    content: [{ type: "text", text: JSON.stringify(responseBody, null, 2) }],
  };
}

const server = new McpServer({
  name: "kokoro-tts-mcp",
  version: "0.1.0",
});

server.tool(
  "say",
  "Queue text for Kokoro speech synthesis and playback. Use get_voices first, then pass one voice identifier.",
  {
    voice: z.string().describe("Kokoro voice identifier, for example af_heart or bm_george."),
    text: z.string().describe("Text to convert to speech."),
  },
  async ({ voice, text }) => {
    console.error(`[kokoro-tts-mcp] say: voice=${voice} text="${text.slice(0, 80)}"`);
    return request("POST", "/say", { text, voice });
  },
);

server.tool(
  "get_voices",
  "List all Kokoro voices and the configured default voice from the speech server.",
  {},
  async () => {
    console.error("[kokoro-tts-mcp] get_voices");
    return request("GET", "/voices");
  },
);

server.tool(
  "get_status",
  "Check status of a speech synthesis request. Returns queued/generating/playing/completed/error plus audio file and error details.",
  {
    message_id: z.string().describe("Message ID returned by the say tool."),
  },
  async ({ message_id }) => {
    console.error(`[kokoro-tts-mcp] get_status: message_id=${message_id}`);
    return request("GET", `/status/${encodeURIComponent(message_id)}`);
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
