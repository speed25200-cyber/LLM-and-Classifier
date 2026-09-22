// Etat global de l'interface (runes Svelte 5) : un seul objet, alimente par le WebSocket d'evenements du coeur.
import { api, ApiError, bootFromPage, bootFromTauri, connectEvents, isTauri } from "./api";
import { newAssistant, reduce } from "./transcript";
import type { AssistantItem, CoreState, DownloadJob, Effort, GpuMetrics, PermissionMode, Session, Settings, VoiceRoute } from "./types";

export type View = "chat" | "models" | "settings";
export interface Toast {
  id: number;
  kind: "info" | "ok" | "warn" | "error" | "voice";
  title: string;
  body?: string;
  ttl: number;
}

class AppState {
  coreStatus = $state<"booting" | "starting" | "ready" | "error">("booting");
  coreError = $state("");
  coreLogs = $state<string[]>([]);
  connected = $state(false);
  core = $state<CoreState | null>(null);
  view = $state<View>("chat");
  session = $state<Session | null>(null);
  metrics = $state<GpuMetrics | null>(null);
  downloads = $state<Record<string, DownloadJob>>({});
  toasts = $state<Toast[]>([]);
  paletteOpen = $state(false);
  inspectorOpen = $state(false);
  inspectorFile = $state<string | null>(null);
  sidebarOpen = $state(true);
  wizardOpen = $state(false);
  planMode = $state(false);
  effortOnce = $state<Effort | null>(null);
  composerText = $state("");
  composerFocus = $state(0);
  benchRunning = $state(false);
  filesVersion = $state(0);
  voice = $state({ recording: false, handsfree: false, level: 0, transcript: "", busy: false, speaking: false, route: null as VoiceRoute | null, error: "" });

  private toastId = 0;
  private stateTimer: ReturnType<typeof setTimeout> | undefined;
  onTurnEnd: ((item: AssistantItem) => void) | null = null;

  get settings(): Settings | null {
    return this.core?.settings ?? null;
  }
  get running(): boolean {
    return !!this.session && !!this.core?.running.includes(this.session.id);
  }
  get ready(): boolean {
    const s = this.core?.runtime.state;
    return s === "ready" || s === "degraded";
  }
  get lastAssistant(): AssistantItem | null {
    const t = this.session?.transcript ?? [];
    for (let i = t.length - 1; i >= 0; i--) if (t[i].kind === "assistant") return t[i] as AssistantItem;
    return null;
  }
  get pendingPermission() {
    const a = this.lastAssistant;
    if (!a) return null;
    for (const b of a.blocks) if (b.type === "permission" && !b.decision) return b;
    return null;
  }

  // ---- demarrage --------------------------------------------------------------------------------------------------
  async init() {
    if (isTauri) {
      await bootFromTauri(
        (info) => {
          this.coreStatus = info.status === "ready" ? "ready" : info.status;
          this.coreError = info.error ?? "";
          if (info.logs) this.coreLogs = info.logs.slice(-200);
          if (info.status === "ready") this.connect();
        },
        (line) => (this.coreLogs = [...this.coreLogs.slice(-199), line]),
      );
      return;
    }
    if (!bootFromPage()) {
      this.coreStatus = "error";
      this.coreError = "Jeton d'acces absent : ouvrez l'interface depuis `prophet-studio`.";
      return;
    }
    this.coreStatus = "ready";
    this.connect();
  }

  private disconnect: (() => void) | null = null;
  private connect() {
    this.disconnect?.();
    this.disconnect = connectEvents(
      (e) => this.onEvent(e),
      (c) => (this.connected = c),
    );
  }

  async refreshState() {
    try {
      const s = await api<CoreState>("/api/state");
      this.applyState(s);
    } catch (e) {
      this.toast("error", "Etat du moteur indisponible", String(e));
    }
  }

  private applyState(s: CoreState) {
    this.core = s;
    for (const j of s.downloads) this.downloads[j.id] = j;
    if (!s.settings.onboarding_done) this.wizardOpen = true;
  }

  private refreshSoon() {
    clearTimeout(this.stateTimer);
    this.stateTimer = setTimeout(() => this.refreshState(), 250);
  }

  // ---- evenements ---------------------------------------------------------------------------------------------------
  private onEvent(e: any) {
    switch (e.type) {
      case "hello": {
        const first = !this.core;
        this.applyState(e.state);
        if (first && !this.session && e.state.sessions.length) this.openSession(e.state.sessions[0].id);
        if (this.session) this.openSession(this.session.id, true);
        return;
      }
      case "runtime.status":
        if (this.core) this.core.runtime = e.runtime;
        if (e.runtime.state === "ready") this.toast("ok", "Modeles prets", e.runtime.plan?.title);
        if (e.runtime.state === "error") this.toast("error", "Demarrage impossible", e.runtime.message);
        return;
      case "runtime.server":
        if (this.core) this.core.runtime.servers[e.server.name as "s1" | "s2"] = e.server;
        return;
      case "runtime.degraded":
        this.toast("warn", "Configuration ajustee automatiquement", e.note);
        return;
      case "download.progress": {
        const j: DownloadJob = e.job;
        const prev = this.downloads[j.id];
        this.downloads[j.id] = j;
        if (j.status === "error" && prev?.status !== "error") this.toast("error", `Echec : ${j.label}`, j.error);
        return;
      }
      case "install.changed":
        if (this.core) this.core.installed = e.installed;
        this.refreshSoon();
        return;
      case "settings.changed":
        if (this.core) {
          this.core.settings = e.settings;
          this.core.plan = e.plan;
        }
        return;
      case "metrics":
        this.metrics = e.gpu;
        if (this.core) {
          this.core.runtime.state = e.runtime;
          this.core.running = e.running;
        }
        return;
      case "turn.queued":
        if (this.core && !this.core.running.includes(e.session_id)) this.core.running = [...this.core.running, e.session_id];
        return;
      case "turn.finished":
        if (this.core) this.core.running = this.core.running.filter((s) => s !== e.session_id);
        this.refreshSessions();
        return;
    }
    if (e.session_id && this.session && e.session_id === this.session.id && e.turn_id) {
      let item = this.session.transcript.find((i) => i.kind === "assistant" && i.turn_id === e.turn_id) as AssistantItem | undefined;
      if (!item) {
        this.session.transcript.push(newAssistant(e.turn_id));
        item = this.session.transcript[this.session.transcript.length - 1] as AssistantItem;
      }
      reduce(item, e);
      if (e.type === "turn.end") this.onTurnEnd?.(item);
      if (e.type === "tool.result" && ["write_file", "edit_file", "python", "run_command"].includes(e.name)) this.filesVersion++;
    }
  }

  // ---- notifications ---------------------------------------------------------------------------------------------------
  toast(kind: Toast["kind"], title: string, body?: string, ttl = 4200) {
    const id = ++this.toastId;
    this.toasts = [...this.toasts.slice(-3), { id, kind, title, body, ttl }];
    setTimeout(() => (this.toasts = this.toasts.filter((t) => t.id !== id)), ttl);
  }

  // ---- sessions ------------------------------------------------------------------------------------------------------------
  async refreshSessions() {
    if (!this.core) return;
    try {
      this.core.sessions = await api("/api/sessions");
    } catch {
      /* hors ligne : on garde la liste */
    }
  }

  async openSession(id: string, quiet = false) {
    try {
      const s = await api<Session>(`/api/sessions/${id}`);
      this.session = s;
      this.view = "chat";
      this.planMode = false;
    } catch (e) {
      if (!quiet) this.toast("error", "Session introuvable", String(e));
    }
  }

  async newSession() {
    const s = await api<Session>("/api/sessions", { method: "POST", body: {} });
    this.session = s;
    this.view = "chat";
    this.planMode = false;
    this.composerFocus++;
    await this.refreshSessions();
  }

  async deleteSession(id: string) {
    await api(`/api/sessions/${id}`, { method: "DELETE" });
    if (this.session?.id === id) this.session = null;
    await this.refreshSessions();
  }

  async renameSession(id: string, title: string) {
    await api(`/api/sessions/${id}`, { method: "PATCH", body: { title } });
    if (this.session?.id === id) this.session.title = title;
    await this.refreshSessions();
  }

  // ---- agent ----------------------------------------------------------------------------------------------------------------
  async send(text: string, opts: { permission_mode?: PermissionMode } = {}) {
    text = text.trim();
    if (!text) return;
    if (!this.ready) {
      this.toast("warn", "Les modeles ne sont pas demarres", "Ouvrez l'ecran Modeles pour les installer ou les lancer.");
      this.view = "models";
      return;
    }
    if (!this.session) await this.newSession();
    const s = this.session!;
    const effort = this.effortOnce ?? undefined;
    this.effortOnce = null;
    try {
      const r = await api<{ turn_id: string }>(`/api/sessions/${s.id}/turn`, {
        body: { text, plan_mode: this.planMode, effort, permission_mode: opts.permission_mode },
      });
      if (!s.transcript.some((i) => i.turn_id === r.turn_id && i.kind === "user")) {
        const idx = s.transcript.findIndex((i) => i.turn_id === r.turn_id);
        const user = { kind: "user" as const, text, ts: Date.now() / 1000, turn_id: r.turn_id, plan_mode: this.planMode };
        if (idx >= 0) s.transcript.splice(idx, 0, user);
        else s.transcript.push(user, newAssistant(r.turn_id, this.planMode));
      }
      if (this.core && !this.core.running.includes(s.id)) this.core.running = [...this.core.running, s.id];
      if (s.title === "Nouvelle session") s.title = text.split("\n")[0].slice(0, 64);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : String(e);
      this.toast("error", "Envoi impossible", msg);
      this.composerText = text;
    }
  }

  async cancel() {
    if (this.session) await api(`/api/sessions/${this.session.id}/cancel`, { method: "POST", body: {} });
  }

  async respondPermission(id: string, allow: boolean, remember = false) {
    try {
      await api(`/api/permissions/${id}`, { body: { allow, remember } });
    } catch (e) {
      this.toast("warn", "Demande expiree", String(e));
    }
  }

  // ---- moteur, modeles, reglages ------------------------------------------------------------------------------------------
  async updateSettings(patch: Partial<Settings> | Record<string, unknown>) {
    try {
      const s = await api<Settings>("/api/settings", { method: "PUT", body: patch });
      if (this.core) this.core.settings = s;
      this.refreshSoon();
    } catch (e) {
      this.toast("error", "Reglage refuse", String(e));
    }
  }

  async install(items?: string[]) {
    try {
      const r = await api<{ jobs: DownloadJob[]; errors: Record<string, string> }>("/api/install", { body: { items } });
      for (const j of r.jobs) this.downloads[j.id] = j;
      for (const [id, err] of Object.entries(r.errors)) this.toast("error", `Installation de ${id} impossible`, err, 8000);
      return r;
    } catch (e) {
      this.toast("error", "Installation impossible", String(e));
    }
  }

  async cancelDownloads(group?: string) {
    await api("/api/downloads/cancel", { body: { group } });
  }

  async removeInstalled(id: string) {
    await api(`/api/installed/${id}`, { method: "DELETE" });
    this.refreshSoon();
  }

  async startRuntime() {
    await api("/api/runtime/start", { method: "POST", body: {} });
  }

  async stopRuntime() {
    await api("/api/runtime/stop", { method: "POST", body: {} });
  }

  async runBench() {
    this.benchRunning = true;
    try {
      const b = await api("/api/bench", { method: "POST", body: {} });
      if (this.core) this.core.bench = b;
      this.toast("ok", "Mesure terminee", `${b.s2_tok_s} tok/s · decisions ${b.s1_p50_ms} ms`);
    } catch (e) {
      this.toast("error", "Mesure impossible", String(e));
    } finally {
      this.benchRunning = false;
    }
  }

  downloadsFor(group: string): DownloadJob[] {
    return Object.values(this.downloads).filter((j) => j.group === group);
  }

  get activeDownloads(): DownloadJob[] {
    return Object.values(this.downloads).filter((j) => ["queued", "running", "verifying", "extracting"].includes(j.status));
  }
}

export const app = new AppState();
