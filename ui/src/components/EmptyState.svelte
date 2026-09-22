<script lang="ts">
  import { AppWindow, FileSearch, Sparkles, Wand2, Mic, Command, AtSign, Slash } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import Logo from "./Logo.svelte";

  const ideas = [
    { icon: AppWindow, title: "Cree une app", text: "Cree une app minuteur Pomodoro en HTML, elegante et sans dependance" },
    { icon: FileSearch, title: "Explore un projet", text: "Explique-moi la structure de ce dossier et ce que fait chaque fichier" },
    { icon: Wand2, title: "Automatise", text: "Ecris un script Python qui renomme mes photos par date de prise de vue" },
    { icon: Sparkles, title: "Reflechis", text: "Compare trois architectures pour une API temps reel et recommande-en une" },
  ];
  const gpu = $derived(app.core?.hardware.primary_gpu?.name?.replace("NVIDIA GeForce ", "") ?? "votre machine");
</script>

<div class="empty">
  <div class="hero rise">
    <div class="mark"><Logo size={56} animated /></div>
    <h1><span class="grad-text">Que construisons-nous ?</span></h1>
    <p class="muted">Bonsai 2 27B raisonne, le classifieur decide en quelques millisecondes. Tout reste sur {gpu}.</p>
  </div>

  {#if !app.ready}
    <div class="callout rise">
      <span class="dot {app.core?.runtime.state === 'starting' ? 'busy' : 'warn'}"></span>
      <span>{app.core?.runtime.state === "starting" ? "Les modeles demarrent…" : "Les modeles ne sont pas encore prets."}</span>
      <button class="btn sm" onclick={() => (app.view = "models")}>Ouvrir les modeles</button>
    </div>
  {/if}

  <div class="ideas">
    {#each ideas as idea, i (idea.title)}
      <button class="idea" style="--d:{i}" onclick={() => app.send(idea.text)}>
        <span class="ic"><idea.icon size={16} /></span>
        <b>{idea.title}</b>
        <span>{idea.text}</span>
      </button>
    {/each}
  </div>

  <div class="hints faint">
    <span><Command size={12} /> <span class="kbd">Ctrl K</span> palette</span>
    <span><Mic size={12} /> <span class="kbd">Ctrl Maj Espace</span> parler</span>
    <span><Slash size={12} /> commandes</span>
    <span><AtSign size={12} /> fichiers</span>
  </div>
</div>

<style>
  .empty { max-width: 780px; margin: 0 auto; padding: 9vh 24px 24px; display: flex; flex-direction: column; align-items: center; gap: 28px; }
  .hero { text-align: center; display: flex; flex-direction: column; align-items: center; gap: 10px; }
  .mark { width: 88px; height: 88px; border-radius: 26px; display: grid; place-items: center; background: radial-gradient(circle at 50% 30%, color-mix(in srgb, var(--s2) 18%, var(--surface-2)), var(--surface)); border: 1px solid var(--line-2); box-shadow: var(--shadow-2); margin-bottom: 8px; }
  h1 { margin: 0; font-size: 34px; letter-spacing: -0.035em; font-weight: 680; line-height: 1.15; }
  .hero p { margin: 0; font-size: 15px; max-width: 520px; }
  .callout { display: flex; align-items: center; gap: 10px; padding: 8px 8px 8px 14px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line-2); font-size: 13px; }
  .ideas { width: 100%; display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
  .idea { text-align: left; display: grid; grid-template-columns: auto 1fr; grid-template-rows: auto auto; column-gap: 12px; row-gap: 2px; padding: 14px 16px; border-radius: 14px; background: var(--surface); border: 1px solid var(--line); transition: border-color var(--t-fast), transform var(--t-med) var(--ease), background var(--t-fast); animation: rise 360ms var(--ease) both; animation-delay: calc(80ms + var(--d) * 50ms); }
  .idea:hover { border-color: var(--line-3); background: var(--surface-2); transform: translateY(-2px); }
  .idea .ic { grid-row: span 2; width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; background: var(--surface-3); color: var(--s2); }
  .idea b { font-size: 13.5px; font-weight: 620; }
  .idea span:last-child { font-size: 12.5px; color: var(--text-3); line-height: 1.45; }
  .hints { display: flex; gap: 18px; flex-wrap: wrap; justify-content: center; font-size: 12px; }
  .hints > span { display: inline-flex; align-items: center; gap: 6px; }
  @media (max-width: 720px) { .ideas { grid-template-columns: 1fr; } }
</style>
