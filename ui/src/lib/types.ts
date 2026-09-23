// Types partages avec le coeur (prophet_studio). Volontairement tolerants : le coeur peut evoluer.

export type RuntimeState = "stopped" | "starting" | "ready" | "degraded" | "error";
export type ServerState = "stopped" | "starting" | "loading" | "ready" | "crashed" | "oom";
export type PermissionMode = "smart" | "ask" | "auto";
export type Effort = "auto" | "fast" | "deep";
export type Priority = "equilibre" | "contexte" | "vitesse";

export interface GPU {
  index: number;
  name: string;
  vendor: string;
  vram_total_mib: number;
  vram_used_mib: number;
  vram_free_mib: number;
  driver: string;
  cuda_version: string;
  compute_cap: string;
  arch: string;
  bandwidth_gbs: number;
  display_active: boolean | null;
  is_blackwell?: boolean;
}

export interface Hardware {
  os: string;
  os_release: string;
  arch: string;
  cpu: string;
  cpu_cores: number;
  cpu_threads: number;
  ram_total_gib: number;
  ram_available_gib: number;
  gpus: GPU[];
  primary_gpu: GPU | null;
  warnings: string[];
}

export interface ServerPlan {
  role: "s1" | "s2";
  model_id: string;
  device: "gpu" | "cpu" | "partial";
  ngl: number;
  ctx: number;
  np: number;
  kv_type: string;
  mmproj: "off" | "cpu" | "gpu";
  reasoning_budget: number;
  threads: number | null;
}

export interface Plan {
  backend: string;
  priority: Priority;
  s2: ServerPlan;
  s1: ServerPlan | null;
  budget: Record<string, number>;
  fits: boolean;
  expected: { s2: { tok_s: [number, number] | null; basis: string }; s1_ms: [number, number] };
  notes: string[];
  rung: number;
  title: string;
}

export interface ServerInfo {
  name: string;
  state: ServerState;
  port: number;
  pid: number | null;
  load_s: number | null;
  error: string;
  argv: string[];
}

export interface Runtime {
  state: RuntimeState;
  message: string;
  mono: boolean;
  plan: Plan | null;
  servers: { s1: ServerInfo; s2: ServerInfo };
}

export interface ModelSpec {
  id: string;
  role: "s1" | "s2" | "voice";
  label: string;
  repo: string;
  size_gb: number;
  quality: string;
  note: string;
  tags: string[];
  params_b: number;
  mmproj_pattern: string;
}

export interface VoiceSpec {
  id: string;
  kind: "stt" | "tts" | "vad";
  label: string;
  size_mb: number;
  languages: string[];
  engine: string;
  note: string;
}

export interface DownloadJob {
  id: string;
  url: string;
  label: string;
  group: string;
  size: number;
  done_bytes: number;
  speed_bps: number;
  status: "queued" | "running" | "verifying" | "extracting" | "done" | "error" | "cancelled";
  error: string;
  eta_s: number | null;
  progress: number | null;
}

export interface VoiceSettings {
  enabled: boolean;
  mode: "ptt" | "handsfree";
  wake_word: string;
  stt_model: string;
  tts_voice: string;
  speak_responses: boolean;
  speed: number;
  command_threshold: number;
}

export interface Settings {
  language: "fr" | "en";
  workspace: string;
  priority: Priority;
  s2_model: string;
  s1_model: string;
  permission_mode: PermissionMode;
  effort: Effort;
  autostart_models: boolean;
  vram_saver: boolean;
  reduce_motion: boolean;
  ctx_override: number;
  llama_server_path: string;
  hf_endpoint: string;
  runtime_tag: string;
  s2_port: number;
  s1_port: number;
  browser_tool: boolean;
  desktop_tool: boolean;
  orca_lora: boolean;
  onboarding_done: boolean;
  voice: VoiceSettings;
}

export interface SessionSummary {
  id: string;
  title: string;
  updated: number;
  workspace: string;
  turns: number;
}

export interface NoulAnswer { noul: number }
export interface Pre {
  direct?: NoulAnswer;
  clarify?: NoulAnswer;
  needs_reasoning?: NoulAnswer;
  risk?: { score: number; probabilities: Record<string, number> };
  intent?: { choice: string; probabilities: Record<string, number>; confidence: number };
  language?: { choice: string; probabilities: Record<string, number>; confidence: number };
  [k: string]: unknown;
}

export interface S1Info {
  pre?: Pre;
  latency_ms?: number;
  budget?: number;
  risk_level?: number;
  path?: "direct" | "agent";
  tools?: Record<string, number>;
}

export type Block =
  | { type: "thinking"; text: string; started?: number; ms?: number }
  | { type: "text"; text: string }
  | {
      type: "tool";
      id: string;
      name: string;
      args: Record<string, unknown>;
      status: "running" | "done" | "error";
      ok?: boolean;
      result?: any;
      ui?: { diff?: string; created?: boolean; lines?: number } | null;
      started?: number;
      ended?: number;
    }
  | {
      type: "permission";
      id: string;
      tool: string;
      describe: string;
      preview: { diff?: string; command?: string; code?: string; old?: string; new?: string } | null;
      judged: Record<string, any> | null;
      decision: "allow" | "deny" | null;
    };

export interface TurnStats {
  s1_ms?: number;
  llm_calls?: number;
  tokens?: number;
  tok_s?: number | null;
  prompt_ms?: number;
  ctx_tokens?: number;
}

export interface AssistantItem {
  kind: "assistant";
  turn_id: string;
  blocks: Block[];
  status: "running" | "done" | "error";
  ts: number;
  s1?: S1Info;
  path?: "direct" | "agent";
  response?: string;
  verification?: number | null;
  latency_ms?: number;
  stopped_by?: string;
  stats?: TurnStats;
  error?: string;
  plan_mode?: boolean;
  pending_tool?: string | null;
}

export interface UserItem {
  kind: "user";
  text: string;
  ts: number;
  turn_id: string;
  plan_mode?: boolean;
}

export type TranscriptItem = UserItem | AssistantItem;

export interface Session {
  id: string;
  title: string;
  workspace: string;
  created: number;
  updated: number;
  transcript: TranscriptItem[];
  always_allow: string[];
  running?: boolean;
}

export interface GpuMetrics {
  util: number;
  vram_used_mib: number;
  vram_total_mib: number;
  temp_c?: number;
  power_w?: number;
  vram_ours_mib?: number;
  simulated?: boolean;
}

export interface Bench {
  s1_p50_ms: number;
  s1_p95_ms: number;
  s2_tok_s: number;
  s2_prefill_tok_s: number;
  s2_total_s: number;
  plan: string | null;
  gpu: string;
  demo: boolean;
  ts: number;
}

export interface CoreState {
  version: string;
  demo: boolean;
  hardware: Hardware;
  settings: Settings;
  plan: Plan;
  runtime: Runtime;
  installed: {
    models: Record<string, { installed: boolean; path: string | null; custom?: boolean; label?: string; role?: string; size_gb?: number }>;
    runtime: { tag: string; backend: string; cuda: number | null; version: string } | null;
    voice: Record<string, boolean>;
    free_disk_gb: number;
    custom_server: boolean;
  };
  downloads: DownloadJob[];
  voice: { engine: string | null; stt_installed: boolean; tts_installed: boolean; vad_installed: boolean };
  sessions: SessionSummary[];
  permissions: { id: string; session_id: string; tool: string }[];
  running: string[];
  recommended: { id: string; label: string; size_gb: number; role?: string }[];
  catalog: { models: ModelSpec[]; voice: VoiceSpec[] };
  bench: Bench | null;
  data_dir: string;
  desktop?: { available: boolean; reason: string };
}

export interface VoiceRoute {
  kind: "command" | "prompt" | "ignored";
  text: string;
  command: string | null;
  confidence: number;
  source: string;
  ms: number;
}

// ---- classifieur (System One) : GGUF importes et calibration par modele ------------------------------------------------
// Fusion de declarations : le catalogue liste aussi les GGUF importes (clone entraine), marques `custom`.
export interface ModelSpec {
  custom?: boolean;
  path?: string;
}

export interface CalibrationReport {
  n: number;
  accuracy: number;
  nll: number;
  brier: number;
  ece: number;
  mean_confidence: number;
}

/** Une calibration par id de modele (`installed.calibration`) ; `error` = fichier illisible, jamais applique. */
export interface S1Calibration {
  model_id: string;
  source?: "studio" | "cli" | "import" | string;
  n?: number | null;
  ts?: number;
  temperature?: Record<string, number>;
  thresholds?: Record<string, number | null>;
  report?: Record<string, { before: CalibrationReport; after: CalibrationReport }>;
  error?: string;
}

export interface S1CalibrationProgress {
  model: string;
  done: number;
  total: number;
}
