/**
 * Hypatia Memory Plugin for OpenCode
 *
 * Hooks into OpenCode's plugin system to automatically log every conversation
 * turn to hypatia knowledge graph.
 *
 * Hooks used:
 * - chat.message  → log user messages
 * - event         → log assistant messages + session lifecycle
 */

import type { Plugin, PluginInput, Hooks } from "@opencode-ai/plugin";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

const execFileAsync = promisify(execFile);

// ── Configuration ──────────────────────────────────────────────

const STATE_DIR = path.join(os.homedir(), ".opencode", "hypatia-memory");
const STATE_FILE = path.join(STATE_DIR, "state.json");
const EXTRACT_SIGNAL = path.join(STATE_DIR, "extract-needed");
const EXTRACT_INTERVAL = 16;

// ── Types ──────────────────────────────────────────────────────

type SessionState = {
  turn: number;
  lastExtract: number;
  project: string;
  /** Most recent user prompt — used to classify intent for memory shaping */
  lastUserText: string;
};

type StateData = Record<string, SessionState>;

// ── State management ───────────────────────────────────────────

async function loadState(): Promise<StateData> {
  try {
    const raw = await fs.readFile(STATE_FILE, "utf-8");
    return JSON.parse(raw);
  } catch {
    return {};
  }
}

async function saveState(state: StateData): Promise<void> {
  await fs.mkdir(STATE_DIR, { recursive: true });
  await fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2), "utf-8");
}

function deriveProject(dir: string): string {
  // Use the last path component as project name
  const parts = dir.replace(/\/+$/, "").split("/");
  return parts[parts.length - 1] || "unknown";
}

// ── Hypatia CLI helpers ────────────────────────────────────────

async function hypatia(args: string[]): Promise<string> {
  try {
    const { stdout } = await execFileAsync("hypatia", args, {
      timeout: 5000,
      env: { ...process.env, PATH: process.env.PATH },
    });
    return stdout.trim();
  } catch (err: any) {
    // Ignore errors in background logging
    if (err.killed) return "";
    return "";
  }
}

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + "...";
}

function redactSecrets(text: string): string {
  // Redaction policy: never store credentials (user rule). Covers API
  // keys, bearer tokens, passwords, private keys, and common platform
  // token prefixes.
  return text
    .replace(/sk-[a-zA-Z0-9]{20,}/g, "sk-***REDACTED***")
    .replace(/Bearer\s+[a-zA-Z0-9._\-]{16,}/g, "Bearer ***REDACTED***")
    .replace(/api[Kk]ey[=:]\s*["']?[a-zA-Z0-9._\-]{16,}["']?/g, "apiKey=***REDACTED***")
    .replace(
      /(password|passwd|pwd|secret|token|access[_-]?key|client[_-]?secret)[=:]\s*["']?[^\s"'，。]{4,}["']?/gi,
      "$1=***REDACTED***"
    )
    .replace(/\b(AKIA|ASIA)[A-Z0-9]{16}\b/g, "***REDACTED-AWS-KEY***")
    .replace(/\bgh[pousr]_[A-Za-z0-9]{20,}\b/g, "***REDACTED-GITHUB-TOKEN***")
    .replace(/\bglpat-[A-Za-z0-9\-_]{15,}\b/g, "***REDACTED-GITLAB-TOKEN***")
    .replace(/\bxox[baprs]-[A-Za-z0-9\-]{10,}\b/g, "***REDACTED-SLACK-TOKEN***")
    .replace(
      /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g,
      "***REDACTED-PRIVATE-KEY***"
    );
}

async function logMessageToHypatia(
  sessionId: string,
  turn: number,
  role: "user" | "assistant",
  content: string,
  project: string
): Promise<void> {
  const name = `msg-${sessionId}-${turn}`;
  // Policy pipeline: redact secrets → absolutize relative dates
  const safeContent = absolutizeDates(
    redactSecrets(truncate(content, 32000)),
    new Date()
  );
  const timestamp = new Date().toISOString();

  const data = `## Role
${role}

## Timestamp
${timestamp}

## Content
${safeContent}`;

  // Create knowledge entry
  await hypatia([
    "knowledge-create", name,
    "-d", data,
    "--tags", "message",
    "--scopes", project,
    "-s", "default",
  ]);

  // Link to session if session entry exists
  // (session entry is created by the AI via the hypatia-memory skill)
}

async function shouldExtract(
  sessionId: string,
  project: string
): Promise<boolean> {
  const state = await loadState();
  const s = state[sessionId];
  if (!s) return false;

  const turnsSince = s.turn - s.lastExtract;
  return turnsSince >= EXTRACT_INTERVAL;
}

async function markExtracted(sessionId: string): Promise<void> {
  const state = await loadState();
  if (state[sessionId]) {
    state[sessionId].lastExtract = state[sessionId].turn;
    await saveState(state);
  }
}

// ── Assistant message accumulation ─────────────────────────────
//
// OpenCode streams assistant content as separate `message.part.updated`
// events (text parts + tool parts). We accumulate them per message and
// only write to hypatia when `message.updated` reports time.completed.

type ToolCallRecord = {
  tool: string;
  callKey: string; // tool + stable input digest, for repeat detection
  inputPreview: string;
  status: "success" | "error";
  error?: string; // cleaned description only — never a stack trace
  durationMs: number;
};

type MessageAccum = {
  texts: Map<string, string>; // partID -> latest text
  tools: Map<string, ToolCallRecord>; // partID -> terminal tool record
  modelID?: string;
  tokens?: { input?: number; output?: number };
};

// In-memory accumulators: sessionID -> messageID -> accum
const accumulators = new Map<string, Map<string, MessageAccum>>();

// Message IDs already written to hypatia (message.updated can re-fire
// after completion, e.g. token/cost updates — never log twice)
const loggedMessages = new Set<string>();
const LOGGED_MAX = 4096;
function markLogged(id: string): void {
  if (loggedMessages.size >= LOGGED_MAX) {
    // Drop the oldest half to bound memory
    const drop = Array.from(loggedMessages).slice(0, LOGGED_MAX >> 1);
    for (const d of drop) loggedMessages.delete(d);
  }
  loggedMessages.add(id);
}

function getAccum(sessionId: string, messageId: string): MessageAccum {
  let perSession = accumulators.get(sessionId);
  if (!perSession) {
    perSession = new Map();
    accumulators.set(sessionId, perSession);
  }
  let acc = perSession.get(messageId);
  if (!acc) {
    acc = { texts: new Map(), tools: new Map() };
    perSession.set(messageId, acc);
  }
  return acc;
}

function dropAccum(sessionId: string, messageId: string): void {
  accumulators.get(sessionId)?.delete(messageId);
  if (accumulators.get(sessionId)?.size === 0) {
    accumulators.delete(sessionId);
  }
}

function stableInputDigest(input: unknown): { key: string; preview: string } {
  let serialized = "";
  try {
    serialized = JSON.stringify(input ?? {}) ?? "";
  } catch {
    serialized = String(input ?? "");
  }
  let preview = serialized;
  if (preview.length > 200) preview = preview.slice(0, 200) + "…";
  // Dependency-free rolling hash over the FULL serialized input
  // (preview alone is too short to be a stable identity).
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < serialized.length && i < 4096; i++) {
    h1 = ((h1 ^ serialized.charCodeAt(i)) * 16777619) >>> 0;
    h2 = (h2 + serialized.charCodeAt(i) * (i + 1)) >>> 0;
  }
  return { key: `${h1.toString(36)}-${h2.toString(36)}`, preview };
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

/**
 * Extract a one-line-ish error DESCRIPTION from a raw error string,
 * stripping stack traces (user rule: no detailed stacks in memory).
 * Handles JS/TS stacks ("    at fn (file:1:2)"), Python tracebacks,
 * Rust panics ("stack backtrace:"), goroutine dumps, etc.
 */
/**
 * One-line brief for native crash dumps that carry no real message
 * text (user rule, e.g. Windows access-violation dumps): keep only the
 * crash kind, drop addresses/frames entirely.
 */
const CRASH_KINDS: Array<[RegExp, string]> = [
  [/0xC0000005|access violation|EXCEPTION_ACCESS_VIOLATION|SIGSEGV|segfault/i, "内存访问违例 (access violation)"],
  [/0xC00000FD|stack overflow|EXCEPTION_STACK_OVERFLOW/i, "栈溢出 (stack overflow)"],
  [/0xC0000409|fast fail|__fastfail/i, "快速失败 (fast fail)"],
  [/0x80000003|breakpoint|SIGTRAP/i, "断点/陷阱异常 (breakpoint)"],
  [/SIGABRT|abort\(\)|0xC000013A/i, "程序主动中止 (abort)"],
  [/SIGBUS/i, "总线错误 (bus error)"],
  [/out of memory|0xC00000FD?|Cannot allocate memory/i, "内存不足 (OOM)"],
  [/deadlock|RtlpWaitOnCriticalSection|hang/i, "疑似死锁/挂起 (deadlock/hang)"],
];

function briefNativeCrash(text: string): string | null {
  // Only applies when the text is dominated by machine details:
  // several hex addresses or module!symbol frames, little prose.
  const addrCount = (text.match(/0x[0-9A-Fa-f]{6,}/g) ?? []).length;
  const frameCount = (text.match(/\b[\w.\-]+![\w.$<>]+\b/g) ?? []).length;
  const wordCount = text.split(/\s+/).filter((w) => /^[a-zA-Z\u4e00-\u9fff]{3,}$/.test(w)).length;
  const machiney = addrCount >= 2 || frameCount >= 2 || (addrCount >= 1 && wordCount <= 6);
  if (!machiney) return null;

  for (const [re, label] of CRASH_KINDS) {
    if (re.test(text)) return `原生崩溃: ${label}（无有效错误消息，地址与堆栈细节已省略）`;
  }
  // Machiney but unrecognized: still summarize, keep a short fragment
  // only if something word-like survives the scrubbing
  const firstFragment = text
    .replace(/0x[0-9A-Fa-f]{2,}/g, "")
    .replace(/\b[\w.\-]+![\w.$<>]+\b/g, "")
    .replace(/[^A-Za-z0-9\u4e00-\u9fff .:'\-()]/g, " ")
    .replace(/\(\s*\)/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const hasWords = /[A-Za-z\u4e00-\u9fff]{4,}/.test(firstFragment);
  return `原生异常（地址与堆栈已省略）${hasWords ? `: ${truncate(firstFragment, 120)}` : "，无有效错误消息"}`;
}

function cleanErrorDescription(raw: string): string {
  let text = (raw ?? "").trim();
  if (!text) return "unknown error";

  // Cut at the first stack marker, keeping any message line before it
  const stackMarkers = [
    /\n\s+at\s/.source, // JS/TS stack frames
    /^Traceback \(most recent call last\):/m.source,
    /stack backtrace:/i.source,
    /^goroutine \d+ /m.source,
    /^\s+File "/m.source, // Python traceback frames
    /\n\s+---+\s*\n?$/m.source,
    /^note: run with /m.source,
  ];
  const combined = new RegExp(stackMarkers.join("|"), "m");
  const m = text.match(combined);
  if (m && m.index !== undefined) text = text.slice(0, m.index);

  let lines = text.split("\n").map((l) => l.trim()).filter(Boolean);

  // Python tracebacks: the exception message is the LAST line — if the
  // cut left nothing (marker at index 0), recover it from the tail.
  if (lines.length === 0) {
    lines = raw
      .split("\n")
      .map((l) => l.trim())
      .filter(
        (l) =>
          l &&
          !/^(Traceback|note: run|goroutine|stack backtrace:|File "|runtime\.|thread panicked)/i.test(l) &&
          !/^\s+at\s/.test(l)
      )
      .slice(-1); // the final exception line
  }

  // Prefer the last non-empty "Error:"-style line from multi-line messages
  const errLines = lines.filter((l) =>
    /^(error|failed|failure|exception|panic|fatal|[A-Za-z_.]+(Error|Exception):)/i.test(l)
  );
  if (errLines.length > 0 && lines.length > 1) {
    text = errLines[errLines.length - 1];
  } else {
    text = lines.join(" ");
  }

  text = text.replace(/\s+/g, " ").trim();
  if (!text) return "unknown error";
  // Native crash dumps with no message text → one-line brief
  const brief = briefNativeCrash(text);
  if (brief) return brief;
  return truncate(text, 300);
}

/**
 * Canonical tool-call ledger (user rule): for every external call
 * (tool/bash/mcp) record WHAT was called, HOW LONG, and WHETHER it
 * succeeded; on failure keep only a cleaned error DESCRIPTION — never
 * stack traces. Repeated identical calls collapse into one entry with
 * a repeat count.
 */
function formatToolLedger(acc: MessageAccum): string {
  if (acc.tools.size === 0) return "";

  const groups = new Map<
    string,
    { rec: ToolCallRecord; recs: ToolCallRecord[]; durations: number[] }
  >();
  for (const rec of acc.tools.values()) {
    const k = `${rec.tool}::${rec.callKey}`;
    const g = groups.get(k);
    if (g) {
      g.recs.push(rec);
      g.durations.push(rec.durationMs);
    } else {
      groups.set(k, { rec, recs: [rec], durations: [rec.durationMs] });
    }
  }

  const lines: string[] = ["## Tool Calls"];
  let i = 0;
  for (const g of groups.values()) {
    i += 1;
    const totalMs = g.durations.reduce((a, b) => a + b, 0);
    const ok = g.recs.filter((r) => r.status === "success");
    const err = g.recs.filter((r) => r.status === "error");
    let line: string;
    if (g.recs.length === 1) {
      line = `${i}. \`${g.rec.tool}\` — ${err.length ? "❌" : "✅"} ${formatDuration(totalMs)}`;
    } else {
      line = `${i}. \`${g.rec.tool}\` ×${g.recs.length} (总用时 ${formatDuration(totalMs)}) — ${ok.length}✅${err.length ? `/${err.length}❌` : ""}`;
    }
    if (err.length === 1) line += ` — ${err[0].error}`;
    else if (err.length > 1) line += ` — 错误: ${err[0].error}（及另外 ${err.length - 1} 次类似失败）`;
    if (g.rec.inputPreview) line += `\n   - 调用: \`${g.rec.inputPreview}\``;
    lines.push(line);
  }
  return lines.join("\n");
}
function buildAssistantMarkdown(acc: MessageAccum): string {
  const sections: string[] = [];

  const body = Array.from(acc.texts.values())
    .map((t) => t.trim())
    .filter(Boolean)
    .join("\n\n");
  if (body) sections.push(body);

  const ledger = formatToolLedger(acc);
  if (ledger) sections.push(ledger);

  const meta: string[] = [];
  if (acc.modelID) meta.push(`model: ${acc.modelID}`);
  if (acc.tokens) meta.push(`tokens: ${acc.tokens.input ?? 0}+${acc.tokens.output ?? 0}`);
  if (sections.length === 0) {
    // Nothing accumulated (e.g. events missed) — keep a minimal marker
    return `_(no content captured)_ ${meta.join(", ")}`.trim();
  }
  if (meta.length) sections.push(`---\n_${meta.join(", ")}_`);
  return sections.join("\n\n");
}

// ── Intent classification & content shaping ────────────────────
//
// Memory policy (user-defined): shape the saved content by what the
// user asked for, not by what the assistant produced:
//   * report request     → save a SUMMARY of the report
//   * discussion         → save markdown context
//   * operation task     → record duration, method, and outcome only
// All saved content gets relative→absolute date conversion and
// secret redaction.

type Intent = "report" | "discussion" | "operation";

const REPORT_RE =
  /(报告|报表|数据分析|分析报告|统计|汇总|总结一下|总结下|简报|dashboard|report|analysis|analyze|summar(?:y|ize|ise)|statistics|insights)/i;
const OPERATION_RE =
  /(运行|执行|跑一下|跑下|安装|部署|发布|修复|修改|创建|新建|删除|清理|编译|提交|推送|启动|停止|重启|配置|设置一下|build|run|exec|install|deploy|publish|fix|repair|create|delete|remove|clean|compile|commit|push|start|stop|restart|set ?up|migrate|refactor)/i;

function classifyIntent(userText: string, hadToolCalls: boolean): Intent {
  const t = userText || "";
  if (REPORT_RE.test(t) && !OPERATION_RE.test(t)) return "report";
  if (OPERATION_RE.test(t)) return "operation";
  // No textual signal: if the assistant had to operate tools, it was a task
  if (hadToolCalls) return "operation";
  return "discussion";
}

/** Summary of a data-analysis report: headings + opening + conclusions. */
function summarizeReport(body: string): string {
  const lines = body.split("\n");
  const headings = lines.filter((l) => /^#{1,4}\s/.test(l.trim()));
  const paras = body
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter(Boolean);
  const chunks: string[] = [];
  if (headings.length) chunks.push(`### Structure\n${headings.join("\n")}`);
  if (paras.length) {
    chunks.push(`### Opening\n${truncate(paras[0], 500)}`);
    if (paras.length > 1) {
      // Conclusions/next-steps live at the end of reports
      chunks.push(`### Conclusion\n${truncate(paras[paras.length - 1], 500)}`);
    }
  }
  return chunks.length ? chunks.join("\n\n") : truncate(body, 800);
}

/** Operation ledger: duration, method, result. */
function buildOperationRecord(
  acc: MessageAccum,
  wallMs: number
): string {
  const lines: string[] = ["## Operation"];

  // Duration
  lines.push(`- 用时: ${formatDuration(wallMs || 0)} (wall)`);

  // Method: which tools were used, how many times
  if (acc.tools.size > 0) {
    const byTool = new Map<string, number>();
    for (const rec of acc.tools.values()) {
      byTool.set(rec.tool, (byTool.get(rec.tool) ?? 0) + 1);
    }
    const method = Array.from(byTool.entries())
      .map(([t, n]) => (n > 1 ? `\`${t}\`×${n}` : `\`${t}\``))
      .join(", ");
    lines.push(`- 手段: ${method}`);
  } else {
    lines.push("- 手段: 无外部工具调用");
  }

  // Result: final assistant statement (last text part), else tool outcomes
  const texts = Array.from(acc.texts.values())
    .map((t) => t.trim())
    .filter(Boolean);
  const outcome =
    texts.length > 0
      ? texts[texts.length - 1]
      : acc.tools.size === 0
        ? "(no outcome text)"
        : null;
  if (outcome !== null) {
    lines.push(`- 结果: ${truncate(outcome, 600)}`);
  } else {
    const ok = Array.from(acc.tools.values()).filter((r) => r.status === "success").length;
    const err = Array.from(acc.tools.values()).filter((r) => r.status === "error").length;
    lines.push(`- 结果: ${ok} 个工具调用成功${err ? `，${err} 个失败` : ""}`);
  }

  return lines.join("\n");
}

/**
 * Replace relative date/time expressions with absolute ones, computed
 * against the actual write time.
 */
function absolutizeDates(text: string, now: Date): string {
  const y = now.getFullYear(),
    m = String(now.getMonth() + 1).padStart(2, "0"),
    d = String(now.getDate()).padStart(2, "0");
  const hm = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;
  const today = `${y}-${m}-${d}`;
  const fmt = (dt: Date) =>
    `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, "0")}-${String(
      dt.getDate()
    ).padStart(2, "0")}`;
  const daysAgo = (n: number) => fmt(new Date(now.getTime() - n * 86400000));

  let out = text;
  // N 天/小时/分钟前 / N days/hours/minutes ago
  out = out.replace(/(\d+)\s*天前/g, (_, n) => daysAgo(parseInt(n)));
  out = out.replace(/(\d+)\s*小时前/g, (_, n) => {
    const t = new Date(now.getTime() - parseInt(n) * 3600000);
    return `${fmt(t)} ${String(t.getHours()).padStart(2, "0")}:00`;
  });
  out = out.replace(/(\d+)\s*分钟前/g, (_, n) => {
    const t = new Date(now.getTime() - parseInt(n) * 60000);
    return `${fmt(t)} ${String(t.getHours()).padStart(2, "0")}:${String(t.getMinutes()).padStart(2, "0")}`;
  });
  out = out.replace(/(\d+)\s*(?:days?|d)\s+ago/gi, (_, n) => daysAgo(parseInt(n)));
  out = out.replace(/(\d+)\s*hours?\s+ago/gi, (_, n) => {
    const t = new Date(now.getTime() - parseInt(n) * 3600000);
    return `${fmt(t)} ${String(t.getHours()).padStart(2, "0")}:00`;
  });
  // 今天/昨天/前天/明天 / today/yesterday/tomorrow
  out = out.replace(/今天|今日|today/gi, today);
  out = out.replace(/昨天|昨日|yesterday/gi, daysAgo(1));
  out = out.replace(/前天/g, daysAgo(2));
  out = out.replace(/明天|明日|tomorrow/gi, daysAgo(-1));
  // 上周/本周/下周 (approximate to the Monday of that week)
  const monday = new Date(now);
  monday.setDate(now.getDate() - ((now.getDay() + 6) % 7));
  const weekOf = (offsetWeeks: number) => {
    const dt = new Date(monday.getTime() + offsetWeeks * 7 * 86400000);
    return `${fmt(dt)} 那周`;
  };
  out = out.replace(/上周/g, weekOf(-1));
  out = out.replace(/本周|这周/g, weekOf(0));
  out = out.replace(/下周/g, weekOf(1));
  // 刚才/刚刚/现在/目前 / just now / currently
  out = out.replace(/刚才|刚刚/g, `${today} ${hm}`);
  out = out.replace(/现在(?![^，。]*是)|目前|此刻|just now|right now/gi, `${today} ${hm}`);
  return out;
}



const hypatiaMemoryPlugin: Plugin = async (
  input: PluginInput,
  _options?: Record<string, unknown>
): Promise<Hooks> => {
  const project = deriveProject(input.directory);

  // Initialize state directory
  await fs.mkdir(STATE_DIR, { recursive: true });

  return {
    /**
     * chat.message — fires when a user sends a message.
     * Log it to hypatia and increment the turn counter.
     */
    "chat.message": async (msgInput, msgOutput) => {
      try {
        const sessionId = msgInput.sessionID;
        const agent = msgInput.agent;

        // Only log the main agent (not compaction/summary sub-agents)
        if (agent && agent !== "build") return;

        // Extract text from message parts
        const textParts = msgOutput.parts
          .filter((p: any) => p.type === "text" && !p.synthetic && !p.ignored)
          .map((p: any) => p.text)
          .filter(Boolean);

        const text = textParts.join("\n").trim();
        if (!text) return;

        // Update turn counter + remember the prompt for intent classification
        const state = await loadState();
        const s: SessionState = state[sessionId] || {
          turn: 0,
          lastExtract: 0,
          project,
          lastUserText: "",
        };
        s.project = project;
        s.turn += 1;
        s.lastUserText = truncate(text, 2000);
        await saveState({ ...state, [sessionId]: s });

        // Fire-and-forget: log to hypatia
        logMessageToHypatia(sessionId, s.turn, "user", text, project).catch(
          () => {} // silently ignore errors
        );
      } catch {
        // Never let hook errors propagate
      }
    },

    /**
     * event — monitors all OpenCode events.
     * We use it to:
     * 1. Accumulate assistant content parts (text + tool calls)
     * 2. On message completion, log full markdown body + condensed
     *    tool-call ledger to hypatia
     * 3. Track session lifecycle
     */
    event: async (eventInput) => {
      try {
        const evt = eventInput.event;

        // ── Assistant content accumulation ──
        if (evt.type === "message.part.updated") {
          const part = (evt.properties as any).part;
          if (!part) return;

          if (part.type === "text") {
            if (part.synthetic || part.ignored) return;
            const acc = getAccum(part.sessionID, part.messageID);
            acc.texts.set(part.id, part.text ?? "");
            return;
          }

          if (part.type === "tool") {
            const state = part.state;
            if (!state) return;
            const acc = getAccum(part.sessionID, part.messageID);
            // NOTE: SDK terminal statuses are "completed" and "error"
            if (state.status === "completed" || state.status === "error") {
              const { key, preview } = stableInputDigest(state.input);
              const durationMs = Math.max(
                0,
                (state.time?.end ?? 0) - (state.time?.start ?? 0)
              );
              const rec: ToolCallRecord = {
                tool: part.tool,
                callKey: key,
                inputPreview: preview,
                status: state.status === "completed" ? "success" : "error",
                durationMs,
                ...(state.status === "error"
                  ? { error: cleanErrorDescription(state.error) }
                  : {}),
              };
              acc.tools.set(part.id, rec);
            }
            // pending / running states: wait for the terminal update
            return;
          }

          return;
        }

        // ── Assistant message completion → write to hypatia ──
        if (evt.type === "message.updated") {
          const msg = (evt.properties as any).info;
          if (!msg || msg.role !== "assistant") return;

          const sessionId = msg.sessionID;
          const state = await loadState();
          const s = state[sessionId];
          if (!s) return; // No state = session not tracked yet

          // Check if this is a completed message (has time.completed and no error)
          if (msg.error) return;
          if (!msg.time?.completed) return;

          const messageId = msg.id;
          const loggedKey = `${sessionId}:${messageId}`;
          if (loggedMessages.has(loggedKey)) return; // already written
          markLogged(loggedKey);

          const acc = getAccum(sessionId, messageId);
          acc.modelID = msg.modelID;
          acc.tokens = {
            input: msg.tokens?.input,
            output: msg.tokens?.output,
          };

          s.turn += 1;
          await saveState({ ...state, [sessionId]: s });

          // Shape the memory by what the user asked for:
          //   report → summary; discussion → markdown context;
          //   operation → duration + method + outcome
          const wallMs = Math.max(
            0,
            (msg.time?.completed ?? 0) - (msg.time?.created ?? 0)
          );
          const intent = classifyIntent(
            s.lastUserText,
            acc.tools.size > 0
          );
          let markdown: string;
          if (intent === "operation") {
            markdown = buildOperationRecord(acc, wallMs);
          } else if (intent === "report") {
            const body = Array.from(acc.texts.values())
              .map((t) => t.trim())
              .filter(Boolean)
              .join("\n\n");
            markdown = `## Report Summary\n${summarizeReport(body)}`;
          } else {
            markdown = buildAssistantMarkdown(acc);
          }
          // Tool-call ledger applies to EVERY intent (user rule)
          const ledger = formatToolLedger(acc);
          if (ledger && !markdown.includes("## Tool Calls")) {
            markdown = `${markdown}\n\n${ledger}`;
          }
          logMessageToHypatia(sessionId, s.turn, "assistant", markdown, project).catch(
            () => {} // silently ignore errors
          );
          dropAccum(sessionId, messageId);

          // Check if we should trigger extraction
          const turnsSince = s.turn - s.lastExtract;
          if (turnsSince >= EXTRACT_INTERVAL) {
            // Write signal file — the AI checks this and runs extraction
            s.lastExtract = s.turn;
            await saveState({ ...state, [sessionId]: s });
            await fs.writeFile(EXTRACT_SIGNAL, String(s.turn), "utf-8").catch(() => {});
          }
        }

        // ── Part removed → drop from accumulator ──
        if (evt.type === "message.part.removed") {
          const props = evt.properties as any;
          if (props?.sessionID && props?.messageID && props?.partID) {
            const perSession = accumulators.get(props.sessionID);
            const acc = perSession?.get(props.messageID);
            acc?.texts.delete(props.partID);
            acc?.tools.delete(props.partID);
          }
        }

        // ── Session lifecycle ──
        if (evt.type === "session.deleted") {
          const sessionId = (evt.properties as any).sessionID;
          if (!sessionId) return;

          // Clean up state
          const state = await loadState();
          delete state[sessionId];
          await saveState(state);
          accumulators.delete(sessionId);
        }
      } catch {
        // Never let hook errors propagate
      }
    },
  };
};

export default hypatiaMemoryPlugin;
