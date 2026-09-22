// Client du coeur : REST + WebSocket d'evenements, en mode navigateur (page servie par le coeur, jeton injecte)
// ou dans l'application de bureau Tauri (coeur supervise par le shell Rust, jeton transmis par `core_info`).

type Boot = { token: string; api: string; demo?: boolean };
type CoreInfo = { url: string | null; token: string; status: "starting" | "ready" | "error"; error?: string | null; logs?: string[] };

declare global {
  interface Window {
    __PROPHET__?: Boot;
    __TAURI__?: any;
  }
}

export const isTauri = typeof window !== "undefined" && !!window.__TAURI__;

let base = "";
let token = "";

export function configure(b: string, t: string) {
  base = b.replace(/\/$/, "");
  token = t;
}

export function coreBase() {
  return base;
}

export function coreToken() {
  return token;
}

/** Mode navigateur : jeton injecte par le coeur (ou VITE_PROPHET_TOKEN en developpement). */
export function bootFromPage(): boolean {
  const b = window.__PROPHET__;
  if (b?.token) {
    configure(b.api || "", b.token);
    return true;
  }
  const dev = import.meta.env.VITE_PROPHET_TOKEN as string | undefined;
  if (dev) {
    configure("", dev);
    return true;
  }
  return false;
}

/** Mode bureau : attend que le shell Tauri ait demarre le coeur. `onStatus` recoit chaque changement d'etat. */
export async function bootFromTauri(onStatus: (info: CoreInfo) => void, onLog?: (line: string) => void): Promise<void> {
  const T = window.__TAURI__;
  const apply = (info: CoreInfo) => {
    if (info.status === "ready" && info.url) configure(info.url, info.token);
    onStatus(info);
  };
  await T.event.listen("core://status", (e: { payload: CoreInfo }) => apply(e.payload));
  if (onLog) await T.event.listen("core://log", (e: { payload: { line: string } }) => onLog(e.payload.line));
  apply(await T.core.invoke("core_info"));
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

export async function api<T = any>(path: string, opts: { method?: string; body?: unknown; raw?: BodyInit; signal?: AbortSignal; headers?: Record<string, string> } = {}): Promise<T> {
  const headers: Record<string, string> = { "X-Prophet-Token": token, ...(opts.headers ?? {}) };
  let body: BodyInit | undefined = opts.raw;
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.body);
  }
  const r = await fetch(base + path, { method: opts.method ?? (body ? "POST" : "GET"), headers, body, signal: opts.signal });
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* corps non JSON */
    }
    throw new ApiError(r.status, msg);
  }
  const ct = r.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return (await r.json()) as T;
  return (await r.blob()) as unknown as T;
}

export function wsUrl(path: string): string {
  const origin = base || location.origin;
  return origin.replace(/^http/, "ws") + path + (path.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
}

/** WebSocket d'evenements avec reconnexion (backoff plafonne a 5 s). */
export function connectEvents(onEvent: (e: any) => void, onState: (connected: boolean) => void): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let delay = 250;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const open = () => {
    ws = new WebSocket(wsUrl("/api/events"));
    ws.onopen = () => {
      delay = 250;
      onState(true);
    };
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data));
      } catch {
        /* message illisible : ignore */
      }
    };
    ws.onclose = () => {
      onState(false);
      if (!closed) timer = setTimeout(open, (delay = Math.min(5000, delay * 2)));
    };
    ws.onerror = () => ws?.close();
  };
  open();
  return () => {
    closed = true;
    clearTimeout(timer);
    ws?.close();
  };
}

/** Controles de fenetre (barre de titre personnalisee de l'application de bureau). */
export const windowControls = {
  minimize: () => window.__TAURI__?.window.getCurrentWindow().minimize(),
  toggleMaximize: () => window.__TAURI__?.window.getCurrentWindow().toggleMaximize(),
  close: () => window.__TAURI__?.window.getCurrentWindow().close(),
};

export async function onTauriEvent(name: string, cb: (payload: any) => void): Promise<void> {
  if (!isTauri) return;
  await window.__TAURI__.event.listen(name, (e: { payload: any }) => cb(e.payload));
}
