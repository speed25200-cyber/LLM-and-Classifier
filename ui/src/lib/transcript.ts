// Reducteur de transcription : miroir exact de prophet_studio/sessions.py:reduce_event, plus quelques champs
// d'affichage (horodatages des outils, outil en preparation, duree de reflexion).
import type { AssistantItem, Block } from "./types";
import type { ComputerAction, ComputerLive, ComputerProgress, ToolBlock } from "./types";

export function newAssistant(turn_id: string, plan_mode = false): AssistantItem {
  return { kind: "assistant", turn_id, blocks: [], status: "running", ts: Date.now() / 1000, plan_mode, pending_tool: null };
}

/** Applique un evenement du tour ; faux si deja applique (tseq : copie vivante relue apres un rechargement). */
export function reduce(item: AssistantItem, evt: any): boolean {
  if (typeof evt.tseq === "number") {
    if (evt.tseq <= (item.tseq ?? 0)) return false;
    item.tseq = evt.tseq;
  }
  const blocks = item.blocks;
  const ts: number = evt.ts ?? Date.now() / 1000;   // horodatage serveur (s) : memes durees en direct et au rechargement
  const last = blocks[blocks.length - 1];
  switch (evt.type) {
    case "thinking.delta":
    case "text.delta": {
      const kind = evt.type === "thinking.delta" ? "thinking" : "text";
      if (kind === "text") closeThinking(item, ts);
      // jamais fusionne par-dessus une voie directe ecartee : la voie agent commence un bloc a elle
      if (last && last.type === kind && blocks.length > (item.reroute?.at ?? 0)) (last as { text: string }).text += evt.text ?? "";
      else blocks.push(kind === "thinking" ? { type: "thinking", text: evt.text ?? "", started: ts } : { type: "text", text: evt.text ?? "" });
      break;
    }
    case "tool.pending":
      item.pending_tool = evt.name;
      closeThinking(item, ts);
      break;
    case "llm.end":
      closeThinking(item, ts);
      break;
    case "tool.call":
      item.pending_tool = null;
      closeThinking(item, ts);
      blocks.push({ type: "tool", id: evt.id, name: evt.name, args: evt.args ?? {}, status: "running", started: ts });
      break;
    case "tool.result":
      for (let i = blocks.length - 1; i >= 0; i--) {
        const b = blocks[i];
        if (b.type === "tool" && b.id === evt.id) {
          b.status = evt.ok ? "done" : "error";
          b.ok = evt.ok;
          b.result = evt.result;
          b.ui = evt.ui;
          b.ended = ts;
          break;
        }
      }
      break;
    case "permission.request":
      blocks.push({ type: "permission", id: evt.id, tool: evt.tool, describe: evt.describe, preview: evt.preview, judged: evt.judged, decision: null });
      break;
    case "permission.resolved":
      for (const b of blocks) if (b.type === "permission" && b.id === evt.id) b.decision = evt.allow ? "allow" : "deny";
      break;
    case "s1.decision": // calibrated / s1_model : l'etat du classifieur pour CE tour
      item.s1 = { ...(item.s1 ?? {}), pre: evt.pre, latency_ms: evt.latency_ms, budget: evt.budget, risk_level: evt.risk_level, path: evt.path,
                  calibrated: evt.calibrated, s1_model: evt.s1_model, gates: evt.gates };
      break;
    case "s1.tools":
      item.s1 = { ...(item.s1 ?? {}), tools: evt.relevance };
      break;
    case "s1.reroute": // voie directe ecartee : les blocs deja la sont la premiere reponse, remplacee
      item.reroute = { reason: evt.reason, verification: evt.verification ?? null, budget: evt.budget ?? null, at: blocks.length };
      // voie et budget reels du tour ; la decision de S1 reste dans rerouted_from
      item.s1 = { ...(item.s1 ?? {}), rerouted_from: item.s1?.path, path: evt.to ?? "agent", ...(evt.budget != null ? { budget: evt.budget } : {}) };
      break;
    case "turn.end":
      Object.assign(item, {
        path: evt.path,
        response: evt.response,
        verification: evt.verification,
        latency_ms: evt.latency_ms,
        stopped_by: evt.stopped_by,
        stats: evt.stats,
        status: "done",
        pending_tool: null,
      });
      break;
    case "turn.error":
      item.status = "error";
      item.error = evt.error;
      break;
    case "computer.escalate":
    case "computer.think":
    case "computer.action":
    case "computer.step":
      reduceComputer(blocks, evt);
      break;
  }
  return true;
}

// ---- computer use : progression par pas (miroir de prophet_studio/sessions.py:reduce_computer) ----
export const CU_KEEP = 40;
const clip = (v: unknown, n = 200): string | null => ((v as string) || "").slice(0, n) || null;

/** Rattache les evenements computer.* a l'outil en cours du meme nom (Prophet execute ses outils un par un). Les actions
 *  de Bonsai arrivent avant le computer.step de leur pas. */
function reduceComputer(blocks: Block[], evt: any) {
  let b: ToolBlock | undefined;
  for (let i = blocks.length - 1; i >= 0 && !b; i--) {
    const x = blocks[i];
    if (x.type === "tool" && x.status === "running" && x.name === evt.tool) b = x as ToolBlock;
  }
  if (!b) return;
  const cu: ComputerProgress = (b.cu ??= { steps: [], n: 0, escalations: 0, pending: [], live: null });
  const base: ComputerLive = cu.live ?? { kind: "", step: cu.n, why: null, turn: null, action: null, blocked: false };
  if (evt.type === "computer.escalate") {
    cu.escalations = evt.escalations || cu.escalations;
    cu.live = { kind: "escalate", step: evt.step ?? cu.n, why: clip(evt.why), turn: null, action: null, blocked: false };
  } else if (evt.type === "computer.think") {
    cu.live = { ...base, kind: "think", turn: evt.turn || 0 };
  } else if (evt.type === "computer.action") {
    const a: ComputerAction = { action: evt.action ?? null, blocked: !!evt.blocked, ok: !!evt.ok };
    cu.pending.push(a);
    cu.live = { ...base, kind: "action", action: a.action, blocked: a.blocked };
  } else {
    const slow: ComputerAction[] = cu.pending.length ? cu.pending : (evt.slow_actions ?? []).map((x: string) => ({ action: x, blocked: false, ok: null }));
    cu.steps.push({
      step: evt.step ?? cu.n, path: evt.path ?? null, action: evt.action ?? null, p: evt.p ?? null, why: clip(evt.why),
      verify: evt.verify ?? null, error: clip(evt.error), escalations: evt.escalations || 0, slow,
    });
    if (cu.steps.length > CU_KEEP) cu.steps.splice(0, cu.steps.length - CU_KEEP);
    cu.n += 1;
    cu.pending = [];
    cu.live = null;
    cu.escalations = evt.escalations || cu.escalations;
  }
}

/** Fige la duree du dernier bloc de reflexion des que le modele passe a autre chose. */
function closeThinking(item: AssistantItem, ts: number) {
  const b = item.blocks[item.blocks.length - 1];
  if (b && b.type === "thinking" && b.started && b.ms == null) b.ms = (ts - b.started) * 1000;
}

/** Le texte diffuse couvre-t-il deja la reponse finale ? (chemin direct : oui ; agent : le resume de `done`)
 *  Apres une reprise, seul compte le texte de la voie agent : la reponse directe ecartee ne tient pas lieu de reponse. */
export function responseAlreadyShown(item: AssistantItem): boolean {
  const text = item.blocks.slice(item.reroute?.at ?? 0).filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join("").trim();
  return !!text && !!item.response && text.replace(/\s+/g, " ").includes(item.response.trim().replace(/\s+/g, " ").slice(0, 80));
}

// Champs du tour poses par les evenements s1.decision / s1.reroute (fusion de declarations avec ./types)
declare module "./types" {
  interface S1Info {
    calibrated?: boolean;                  // calibration appliquee a CE tour (absent : tour enregistre avant ce champ)
    s1_model?: string | null;              // classifieur qui a lu la demande (null : mode mono)
    gates?: Record<string, number | null>; // seuils calibres des portes (null : ne jamais se fier a cette lecture)
    rerouted_from?: "direct" | "agent";    // decision de S1 avant la reprise en voie agent
  }
  interface RerouteInfo {
    budget?: number | null;                // budget de reflexion reel de la voie agent
  }
}
