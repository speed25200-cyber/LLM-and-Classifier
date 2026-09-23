<script lang="ts">
  import { ChevronDown, Zap } from "@lucide/svelte";
  import type { S1Info } from "../lib/types";
  import { num, pct } from "../lib/format";

  // La signature de la fusion : ce que le classifieur (System One) a decide avant que Bonsai ne pense.
  // rerouted : tour repris en voie agent (repli pour les sessions enregistrees avant `rerouted_from`).
  let { s1, rerouted = false }: { s1: S1Info; rerouted?: boolean } = $props();
  let open = $state(false);
  const pre = $derived(s1.pre ?? {});
  const risk = $derived(s1.risk_level ?? Math.round(pre.risk?.score ?? 0));
  const tools = $derived(Object.entries(s1.tools ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 8));
  const intent = $derived(pre.intent?.choice);
  const INTENT_FR: Record<string, string> = { chat: "conversation", create_app: "creer une app", modify_code: "modifier du code", run_command: "executer", browse: "web", remember: "memoire", other: "autre" };
  // calibration appliquee a CE tour (enregistree avec lui) : undefined pour un tour plus ancien que ce champ -> ni l'un ni l'autre
  const calibrated = $derived(s1.calibrated);
  const fromDirect = $derived(s1.rerouted_from === "direct" || (rerouted && s1.path === "direct"));
  const GATE_FR: Record<string, string> = { direct: "direct", needs_reasoning: "sans reflexion", risk: "risque" };
  const gates = $derived(Object.entries(s1.gates ?? {}));
</script>

<div class="s1">
  {#if pre.s1_error}
    <div class="row"><span class="chip warn"><Zap size={12} /> classifieur indisponible : Bonsai continue seul, chaque action risquee est confirmee</span></div>
  {:else}
  <button class="row" onclick={() => (open = !open)} aria-expanded={open}>
    <span class="chip s1 lead"><Zap size={12} /> System One · {num(s1.latency_ms ?? 0)} ms</span>
    {#if s1.path}
      <span class="chip" style="--d:1" title={fromDirect ? "S1 avait choisi la reponse directe ; elle a ete ecartee et le tour repris avec les outils" : undefined}>
        {fromDirect ? "direct → agent" : s1.path === "direct" ? "reponse directe" : "agent"}
      </span>
    {/if}
    {#if pre.needs_reasoning}
      <span class="chip" style="--d:2">raisonnement {pct(pre.needs_reasoning.noul)}</span>
    {/if}
    <span class="chip {risk >= 2 ? 'warn' : ''}" style="--d:3">risque {risk}/3</span>
    {#if s1.budget != null}
      <span class="chip s2" style="--d:4">reflexion {s1.budget ? `${num(s1.budget)} tok` : "aucune"}</span>
    {/if}
    {#if pre.clarify && pre.clarify.noul >= 0.5}<span class="chip warn" style="--d:5">ambigu {pct(pre.clarify.noul)}</span>{/if}
    {#if s1.gates?.direct === null}
      <span class="chip warn" style="--d:6" title="Calibration de ce classifieur : aucune lecture 'reponse directe' n'atteint la precision visee sur les graines etiquetees, il ne prend donc jamais seul la voie directe. Re-entrainez-le ou recalibrez.">voie directe coupee par la calibration</span>
    {/if}
    {#if calibrated === false}
      <span class="chip" style="--d:6" title={s1.s1_model
        ? `Aucune calibration appliquee a ${s1.s1_model} pour ce tour : pourcentages bruts. Ecran Modeles -> Calibrer.`
        : "Mode mono : Bonsai a repondu aux questions du classifieur, pourcentages bruts."}>non calibre</span>
    {/if}
    <ChevronDown size={14} class="chev {open ? 'open' : ''}" />
  </button>
  {#if open}
    <div class="detail fade-in">
      <div class="col">
        <div class="panel-title">{calibrated === true ? "Jugements calibres" : calibrated === false ? "Jugements (non calibres)" : "Jugements"}</div>
        {#each [["reponse directe", pre.direct?.noul], ["raisonnement", pre.needs_reasoning?.noul], ["question utile", pre.clarify?.noul]] as [label, v]}
          {#if v != null}
            <div class="meter"><span>{label}</span><div class="bar"><i style="transform: scaleX({v})"></i></div><b class="tabnum">{pct(v as number)}</b></div>
          {/if}
        {/each}
        {#if intent}<p class="faint obs">Observation : {INTENT_FR[intent] ?? intent}{pre.language?.choice && pre.language.choice !== "none" ? ` · ${pre.language.choice}` : ""}</p>{/if}
        {#if gates.length}
          <p class="faint obs" title="Seuils de la calibration de ce classifieur : sous le seuil, la lecture n'est pas jugee assez sure pour agir seule">
            Seuils calibres : {gates.map(([k, v]) => `${GATE_FR[k] ?? k} ${v == null ? "jamais" : pct(v)}`).join(" · ")}
          </p>
        {/if}
      </div>
      {#if tools.length}
        <div class="col">
          <div class="panel-title">Pertinence des outils</div>
          {#each tools as [name, v] (name)}
            <div class="meter"><span class="mono">{name}</span><div class="bar"><i style="transform: scaleX({v})"></i></div><b class="tabnum">{pct(v)}</b></div>
          {/each}
        </div>
      {/if}
    </div>
  {/if}
  {/if}
</div>

<style>
  .s1 { margin-bottom: 12px; }
  .row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; text-align: left; }
  .row .chip { animation: rise 300ms var(--ease) both; animation-delay: calc(var(--d, 0) * 45ms); height: 22px; font-size: 11.5px; }
  .lead { font-weight: 620; }
  :global(.s1 .chev) { color: var(--text-3); transition: transform var(--t-med) var(--ease); }
  :global(.s1 .chev.open) { transform: rotate(180deg); }
  .detail { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 18px; margin-top: 10px; padding: 14px 16px; border-radius: 14px; background: var(--surface); border: 1px solid var(--line); }
  .col { display: flex; flex-direction: column; gap: 8px; }
  .meter { display: grid; grid-template-columns: 110px 1fr 44px; align-items: center; gap: 10px; font-size: 12px; color: var(--text-2); }
  .meter .mono { font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; }
  .meter b { font-weight: 560; text-align: right; color: var(--text); font-size: 11.5px; }
  .meter .bar i { background: linear-gradient(90deg, var(--s1), color-mix(in srgb, var(--s1) 50%, var(--s2))); }
  .obs { margin: 4px 0 0; font-size: 12px; }
</style>
