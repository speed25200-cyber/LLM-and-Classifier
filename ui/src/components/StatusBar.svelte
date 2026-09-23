<script lang="ts">
  import { Mic, MicOff, Radio, Thermometer, Zap, Brain, HardDrive, Wifi, WifiOff } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import { ctxLabel, mibToGib, num } from "../lib/format";
  import { toggleHandsfree } from "../lib/voice";
  import type { Runtime } from "../lib/types";

  const plan = $derived(app.core?.runtime.plan ?? app.core?.plan);
  // relance automatique en cours (restarting) et relances deja faites par le watchdog (restarts, au plus une par serveur)
  const rt = $derived(app.core?.runtime as (Runtime & { restarting?: "s1" | "s2" | null }) | undefined);
  const relaunches = $derived.by(() => {
    const r = rt?.restarts;
    return r ? [r.s1 ? `S1 x${r.s1}` : "", r.s2 ? `S2 x${r.s2}` : ""].filter(Boolean).join(" · ") : "";
  });
  const m = $derived(app.metrics);
  const last = $derived.by(() => {
    const t = app.session?.transcript ?? [];
    for (let i = t.length - 1; i >= 0; i--) {
      const it = t[i];
      if (it.kind === "assistant" && (it.stats?.tok_s || it.s1?.latency_ms)) return it;
    }
    return null;
  });
  const dl = $derived(app.activeDownloads);
  const dlProgress = $derived.by(() => {
    const total = dl.reduce((a, j) => a + (j.size || 0), 0);
    return total ? dl.reduce((a, j) => a + j.done_bytes, 0) / total : null;
  });
</script>

<footer class="status tabnum">
  <div class="l">
    <span class="it" title={app.connected ? "Connecte au moteur" : "Moteur injoignable"}>
      {#if app.connected}<Wifi size={12} />{:else}<WifiOff size={12} class="err" />{/if}
    </span>
    {#if plan}
      <span class="it plan" title={plan.notes.join("\n")}>{plan.title}{plan.rung ? " · ajuste" : ""}</span>
      {@const used = last?.stats?.ctx_tokens ?? 0}
      <span class="it" title="Contexte occupe par la conversation / fenetre de Bonsai{used ? ' (au dernier tour)' : ''}">
        ctx {used ? `${ctxLabel(used)} / ` : ""}{ctxLabel(plan.s2.ctx)}
        {#if used}<span class="mini" class:hot={used / plan.s2.ctx > 0.8}><i style="transform: scaleX({Math.min(1, used / plan.s2.ctx)})"></i></span>{/if}
      </span>
    {/if}
    {#if rt?.restarting}
      <span class="it warn" title={rt.message}><span class="spinner"></span> redemarrage {rt.restarting === "s1" ? "du classifieur" : "de Bonsai"}</span>
    {:else if rt?.mono && (rt.state === "ready" || rt.state === "degraded")}
      <span class="it warn" title={rt.message || "Bonsai repond aussi aux decisions System One"}>mode mono</span>
    {/if}
    {#if relaunches && !rt?.restarting}
      <button class="it warn" onclick={() => (app.view = "models")}
        title={`Relances automatiques apres un arret inattendu (au plus une par serveur)${rt?.message ? ` : ${rt.message}` : ""}`}>relance {relaunches}</button>
    {/if}
    {#if dl.length}
      <button class="it dl" onclick={() => (app.view = "models")}>
        <span class="spinner"></span> {dl.length} telechargement{dl.length > 1 ? "s" : ""}{dlProgress != null ? ` · ${Math.round(dlProgress * 100)} %` : ""}
      </button>
    {/if}
  </div>
  <div class="r">
    {#if last?.s1?.latency_ms != null}
      <span class="it s1" title="Derniere decision du classifieur (System One)"><Zap size={12} /> {num(last.s1.latency_ms)} ms</span>
    {/if}
    {#if last?.stats?.tok_s}
      <span class="it s2" title="Vitesse de generation de Bonsai (System Two)"><Brain size={12} /> {num(last.stats.tok_s, 1)} tok/s</span>
    {/if}
    {#if m}
      <span class="it" title="Memoire video utilisee{m.simulated ? ' (simulee en demo)' : ''}">
        <HardDrive size={12} /> {mibToGib(m.vram_used_mib)} / {mibToGib(m.vram_total_mib)}
        <span class="mini"><i style="transform: scaleX({m.vram_used_mib / m.vram_total_mib})"></i></span>
      </span>
      <span class="it" title="Charge GPU">GPU {m.util} %</span>
      {#if m.temp_c != null}<span class="it"><Thermometer size={12} /> {m.temp_c}°</span>{/if}
    {/if}
    <button class="it voice" class:on={app.voice.handsfree} onclick={() => toggleHandsfree()} title="Mains libres (mot d'eveil)">
      {#if app.voice.handsfree}<Radio size={12} /> ecoute{:else if app.voice.recording}<Mic size={12} /> micro{:else}<MicOff size={12} /> voix{/if}
    </button>
  </div>
</footer>

<style>
  .status { height: var(--statusbar-h); flex: none; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 10px; border-top: 1px solid var(--line); font-size: 11.5px; color: var(--text-3); background: var(--bg-2); }
  .l, .r { display: flex; align-items: center; gap: 2px; min-width: 0; }
  .it { display: inline-flex; align-items: center; gap: 5px; height: 22px; padding: 0 7px; border-radius: 6px; white-space: nowrap; }
  button.it:hover { background: var(--surface-2); color: var(--text-2); }
  .plan { overflow: hidden; text-overflow: ellipsis; max-width: 460px; display: inline-block; line-height: 22px; }
  .s1 { color: var(--s1); }
  .s2 { color: var(--s2); }
  .warn { color: var(--warn); }
  .dl { color: var(--text-2); }
  .it .spinner { width: 10px; height: 10px; border-width: 1.5px; }
  .mini { position: relative; display: inline-block; width: 38px; height: 4px; border-radius: 4px; background: var(--surface-3); overflow: hidden; margin-left: 2px; }
  .mini i { position: absolute; inset: 0; background: var(--grad); transform-origin: left; transition: transform 600ms var(--ease); }
  .mini.hot i { background: var(--warn); }
  .voice.on { color: var(--s1); background: var(--s1-soft); }
  :global(.status .err) { color: var(--err); }
</style>
