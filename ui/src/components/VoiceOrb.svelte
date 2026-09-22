<script lang="ts">
  import { app } from "../lib/store.svelte";
  const show = $derived(app.voice.recording || app.voice.busy || (app.voice.handsfree && !!app.voice.transcript));
  const label = $derived(app.voice.busy ? "Transcription…" : app.voice.recording ? (app.voice.handsfree ? "Je vous ecoute…" : "Parlez, relachez pour envoyer") : "");
</script>

{#if show}
  <div class="orbwrap rise" role="status">
    <div class="orb" class:busy={app.voice.busy} style="--l:{app.voice.level}">
      <i class="r1"></i><i class="r2"></i><i class="core"></i>
    </div>
    <div class="txt">
      <b class:shimmer={app.voice.busy}>{label}</b>
      {#if app.voice.transcript && !app.voice.recording}
        <span>« {app.voice.transcript} »{#if app.voice.route?.kind === "command"} → <em>{app.voice.route.command}</em>{/if}</span>
      {/if}
    </div>
  </div>
{/if}

<style>
  .orbwrap { position: fixed; left: 50%; bottom: calc(var(--statusbar-h) + 120px); transform: translateX(-50%); z-index: 40; display: flex; align-items: center; gap: 14px; padding: 10px 20px 10px 12px; border-radius: 999px; background: color-mix(in srgb, var(--surface-2) 94%, transparent); border: 1px solid var(--line-2); box-shadow: var(--shadow-3); max-width: min(620px, 90vw); }
  .orb { position: relative; width: 42px; height: 42px; flex: none; }
  .orb i { position: absolute; inset: 0; border-radius: 50%; }
  .core { background: radial-gradient(circle at 35% 30%, #b8fff4, var(--s1) 45%, #1a8f86); transform: scale(calc(0.72 + var(--l) * 0.28)); transition: transform 90ms linear; }
  .r1 { background: color-mix(in srgb, var(--s1) 28%, transparent); transform: scale(calc(0.9 + var(--l) * 0.7)); transition: transform 120ms linear; }
  .r2 { border: 1.5px solid color-mix(in srgb, var(--s2) 50%, transparent); animation: ring 2.2s var(--ease) infinite; }
  .busy .core { animation: pulse 1s var(--ease) infinite; }
  @keyframes ring { from { transform: scale(0.9); opacity: 0.9; } to { transform: scale(1.6); opacity: 0; } }
  .txt { display: flex; flex-direction: column; min-width: 0; }
  .txt b { font-size: 13px; font-weight: 600; }
  .txt span { font-size: 12.5px; color: var(--text-2); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  em { color: var(--s1); font-style: normal; font-family: var(--mono); font-size: 12px; }
</style>
