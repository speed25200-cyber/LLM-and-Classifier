<script lang="ts">
  import { app } from "../lib/store.svelte";
  import { isTauri } from "../lib/api";
  import Logo from "./Logo.svelte";

  const logs = $derived(app.coreLogs.slice(-8));
  async function restart() {
    await window.__TAURI__?.core.invoke("restart_core");
  }
</script>

<div class="boot">
  <div class="inner rise">
    <Logo size={64} animated />
    {#if app.coreStatus === "error"}
      <h1>Le moteur n'a pas demarre</h1>
      <p class="muted">{app.coreError || "Erreur inconnue."}</p>
      {#if isTauri}<button class="btn primary" onclick={restart}>Reessayer</button>{/if}
    {:else}
      <h1 class="shimmer">Preparation du moteur</h1>
      <p class="muted">{isTauri ? "Premier lancement : installation de Python et des dependances (1 a 3 minutes)." : "Connexion au moteur local…"}</p>
    {/if}
    {#if logs.length}
      <pre class="logs">{logs.join("\n")}</pre>
    {/if}
  </div>
</div>

<style>
  .boot { flex: 1; display: grid; place-items: center; padding: 40px; }
  .inner { display: flex; flex-direction: column; align-items: center; gap: 14px; max-width: 560px; text-align: center; }
  h1 { margin: 10px 0 0; font-size: 22px; letter-spacing: -0.02em; font-weight: 650; }
  p { margin: 0; }
  .logs { margin-top: 12px; width: 100%; max-height: 180px; overflow: hidden; text-align: left; font-size: 11.5px; line-height: 1.55; color: var(--text-3); background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; white-space: pre-wrap; }
</style>
