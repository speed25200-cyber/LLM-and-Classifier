<script module lang="ts">
  export const TOOL_FR: Record<string, string> = { write_file: "ecrire un fichier", edit_file: "modifier un fichier", run_command: "executer une commande",
    python: "executer du Python", create_tool: "creer un nouvel outil", browse: "naviguer sur le web", desktop: "piloter le bureau",
    desktop_step: "action sur le bureau", browse_step: "action dans le navigateur" };
  export const RISK_FR: Record<string, string> = { readonly: "lecture seule", workspace_write: "ecriture locale", destructive: "destructif", privileged: "privilegie", exfiltration: "sortie de donnees" };
  const RISKY = ["destructive", "privileged", "exfiltration"];

  /** Autorisation memorisee qui ne peut plus rien autoriser : classe dangereuse (arret obligatoire), ou pas du computer use
   *  sans classe (ancienne regle : un pas est toujours juge par le clone). Revocable, affichee comme telle. */
  export function grantInert(g: string): boolean {
    const i = g.indexOf(":");
    return i < 0 ? g.endsWith("_step") : RISKY.includes(g.slice(i + 1));
  }

  /** Autorisation memorisee : "run_command:readonly" -> "executer une commande · lecture seule" ; "write_file" -> action non jugee. */
  export function grantLabel(g: string): string {
    const i = g.indexOf(":");
    const tool = i < 0 ? g : g.slice(0, i);
    const cls = i < 0 ? "" : g.slice(i + 1);
    const label = `${TOOL_FR[tool] ?? tool} · ${cls ? (RISK_FR[cls] ?? cls) : "sans jugement"}`;
    return grantInert(g) ? `${label} (sans effet)` : label;
  }
</script>

<script lang="ts">
  import { ShieldAlert, ShieldCheck, ShieldX, Zap } from "@lucide/svelte";
  import type { Block } from "../lib/types";
  import { app } from "../lib/store.svelte";
  import { highlight } from "../lib/markdown";
  import { num, pct } from "../lib/format";
  import DiffView from "./DiffView.svelte";

  type Perm = Extract<Block, { type: "permission" }>;
  let { b }: { b: Perm } = $props();

  const j = $derived(b.judged ?? {});
  const pending = $derived(!b.decision);
  // jugee par le clone ? (anciennes transcriptions : "workspace_write" sans s1_consulted etait une valeur de remplissage)
  const judged = $derived(j.s1_consulted === true || (j.s1_consulted == null && !!j.tool_risk && j.tool_risk !== "workspace_write"));
  // arret obligatoire (danger probable, politique, classifieur en panne, action vue en partie, code invisible) : jamais de « toujours »
  const hard = $derived(!!j.hard_stop);
  // « toujours » seulement si le serveur propose une cle : ni arret obligatoire, ni verdict incertain du clone
  const grantable = $derived(typeof j.grant === "string" && !hard);
  const unsure = $derived(judged && !hard && !grantable && !!j.needs_confirmation);
  const cls = $derived(grantable && j.grant.includes(":") ? j.grant.slice(j.grant.indexOf(":") + 1) : "");

  function decide(allow: boolean, remember = false) {
    if (pending) app.respondPermission(b.id, allow, remember && grantable);
  }

  function onKey(e: KeyboardEvent) {
    if (!pending || app.pendingPermission?.id !== b.id || e.ctrlKey || e.metaKey || e.altKey) return;
    // dans la zone de saisie, les raccourcis ne valent que si elle est vide (sinon on tape normalement)
    const field = (e.target as HTMLElement)?.closest?.("input, textarea");
    if (field && (field.tagName === "INPUT" || app.composerText.trim())) return;
    const k = e.key.toLowerCase();
    if (k === "y" || k === "o") (e.preventDefault(), decide(true));
    else if ((k === "a" || k === "t") && grantable) (e.preventDefault(), decide(true, true));
    else if (k === "n" || k === "r") (e.preventDefault(), decide(false));
  }
</script>

<svelte:window onkeydown={onKey} />

<div class="perm" class:pending class:allowed={b.decision === "allow"} class:denied={b.decision === "deny"}>
  <div class="head">
    <span class="ic">
      {#if b.decision === "allow"}<ShieldCheck size={16} />{:else if b.decision === "deny"}<ShieldX size={16} />{:else}<ShieldAlert size={16} />{/if}
    </span>
    <div class="t">
      <b>{pending ? "Autorisation requise" : b.decision === "allow" ? "Autorise" : "Refuse"} · {TOOL_FR[b.tool] ?? b.tool}</b>
      <div class="chips">
        {#if j.s1_error}
          <span class="chip warn"><Zap size={11} /> classifieur indisponible : confirmation par prudence</span>
        {:else if judged}
          <span class="chip s1"><Zap size={11} /> classifieur : {RISK_FR[j.tool_risk] ?? j.tool_risk}{j.tool_risk_conf != null ? ` (${pct(j.tool_risk_conf)} de certitude)` : ""}</span>
          {#if j.p_risky != null && j.p_risky >= 0.2}<span class="chip {j.p_risky >= 0.35 ? 'warn' : ''}" title="Probabilite destructif + privilegie + sortie de donnees">danger {pct(j.p_risky)}</span>{/if}
          <span class="chip {j.risk >= 2 ? 'warn' : ''}">risque {num(j.risk ?? 0, 1)}/3</span>
          {#if j.policy_violation >= 0.3}<span class="chip warn">politique {pct(j.policy_violation)}</span>{/if}
          {#if j.partial}<span class="chip warn">trop long pour etre juge en entier</span>{/if}
          {#if j.unseen?.length}<span class="chip warn" title={j.unseen.join(", ")}>code lance invisible pour le juge</span>{/if}
          {#if j.meta_skills}<span class="chip warn">vise les outils de Prophet (.prophet/skills)</span>{/if}
          {#if j.latency_ms}<span class="chip">juge en {num(j.latency_ms)} ms</span>{/if}
        {:else}
          <span class="chip">non juge · mode « toujours demander »</span>
        {/if}
        {#if hard && pending}<span class="chip warn" title="Ni « toujours » ni une autorisation memorisee ne s'appliquent">confirmation obligatoire</span>
        {:else if unsure && pending}<span class="chip" title="« Toujours » ne couvre que les verdicts surs du classifieur">verdict incertain</span>{/if}
      </div>
    </div>
  </div>

  {#if pending || b.preview}
    <div class="preview">
      {#if b.preview?.diff}
        <DiffView diff={b.preview.diff} max={120} />
      {:else if b.preview && "diff" in b.preview}
        <div class="same">Contenu identique au fichier existant : aucune modification.</div>
      {:else if b.preview?.old != null && b.preview?.new != null}
        <DiffView oldText={b.preview.old} newText={b.preview.new} max={120} />
      {:else if b.preview?.command}
        <div class="cmd mono"><span class="ps">$</span> {b.preview.command}</div>
      {:else if b.preview?.code}
        <pre class="code mono">{@html highlight(b.preview.code, "python")}</pre>
      {:else}
        <pre class="code mono">{b.describe}</pre>
      {/if}
    </div>
  {/if}

  {#if pending}
    <div class="acts">
      <button class="btn primary sm" onclick={() => decide(true)}>Autoriser <span class="kbd">Y</span></button>
      {#if grantable}
        <button class="btn sm" onclick={() => decide(true, true)} title="Revocable dans le panneau Espace de travail">
          Toujours pour cet outil{cls ? ` (${RISK_FR[cls] ?? cls})` : ""} <span class="kbd">A</span>
        </button>
      {/if}
      <button class="btn ghost sm danger" onclick={() => decide(false)}>Refuser <span class="kbd">N</span></button>
      <span class="hint faint">ou dites « accepte » / « refuse »</span>
    </div>
  {/if}
</div>

<style>
  .perm { margin: 8px 0; border-radius: 14px; border: 1px solid var(--line-2); background: var(--surface); overflow: hidden; animation: rise 280ms var(--ease) both; }
  .perm.pending { border-color: color-mix(in srgb, var(--warn) 45%, transparent); box-shadow: 0 0 0 4px color-mix(in srgb, var(--warn) 8%, transparent), var(--shadow-2); }
  .perm.allowed { border-color: color-mix(in srgb, var(--ok) 25%, var(--line)); }
  .perm.denied { border-color: color-mix(in srgb, var(--err) 25%, var(--line)); opacity: 0.85; }
  .head { display: flex; gap: 12px; padding: 12px 14px; }
  .ic { width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center; flex: none; background: var(--warn-soft); color: var(--warn); }
  .allowed .ic { background: var(--ok-soft); color: var(--ok); }
  .denied .ic { background: var(--err-soft); color: var(--err); }
  .t { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .t b { font-size: 13.5px; font-weight: 620; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chips .chip { height: 21px; font-size: 11px; }
  .preview { border-top: 1px solid var(--line); max-height: 320px; overflow: auto; }
  .cmd { padding: 12px 14px; background: var(--bg-2); font-size: 12.5px; white-space: pre-wrap; word-break: break-all; }
  .ps { color: var(--s1); }
  .same { padding: 12px 14px; font-size: 12.5px; color: var(--text-3); }
  .code { margin: 0; padding: 12px 14px; background: var(--bg-2); font-size: 12px; white-space: pre-wrap; }
  .acts { display: flex; align-items: center; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--line); background: var(--surface-2); flex-wrap: wrap; }
  .acts .kbd { background: transparent; border-color: currentColor; opacity: 0.5; height: 17px; }
  .hint { font-size: 11.5px; margin-left: auto; }
</style>
