<script lang="ts">
  import { Command, Minus, PanelLeft, Square, X, Moon, Sun } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import { isTauri, windowControls } from "../lib/api";
  import Logo from "./Logo.svelte";

  let { onTheme, theme }: { onTheme: () => void; theme: string } = $props();
  const mac = /Mac/i.test(navigator.userAgent);
  let editing = $state(false);
  let draft = $state("");

  const title = $derived(app.view === "models" ? "Modeles et materiel" : app.view === "settings" ? "Reglages" : (app.session?.title ?? "Prophet Studio"));
  const rt = $derived(app.core?.runtime);
  const s2label = $derived.by(() => {
    const id = rt?.plan?.s2.model_id ?? app.core?.plan.s2.model_id;
    return app.core?.catalog.models.find((m) => m.id === id)?.label ?? "Bonsai";
  });

  function startEdit() {
    if (app.view !== "chat" || !app.session) return;
    draft = app.session.title;
    editing = true;
  }
  async function commit() {
    editing = false;
    if (app.session && draft.trim() && draft !== app.session.title) await app.renameSession(app.session.id, draft.trim());
  }
</script>

<header class="bar" class:mac={mac && isTauri} data-tauri-drag-region>
  <div class="left" data-tauri-drag-region>
    <button class="icon-btn" title="Barre laterale (Ctrl \)" onclick={() => (app.sidebarOpen = !app.sidebarOpen)}><PanelLeft size={16} /></button>
    <div class="brand" data-tauri-drag-region>
      <Logo size={18} thinking={app.running} />
      <span data-tauri-drag-region>Prophet <b>Studio</b></span>
      {#if app.core?.demo}<span class="chip warn demo">demo</span>{/if}
    </div>
  </div>

  <div class="center" data-tauri-drag-region>
    {#if editing}
      <!-- svelte-ignore a11y_autofocus -->
      <input class="title-input" bind:value={draft} autofocus onblur={commit} onkeydown={(e) => (e.key === "Enter" ? commit() : e.key === "Escape" && (editing = false))} />
    {:else}
      <button class="title" ondblclick={startEdit} title="Double-clic pour renommer">{title}</button>
    {/if}
  </div>

  <div class="right" data-tauri-drag-region>
    <button class="palette" onclick={() => (app.paletteOpen = true)} title="Palette de commandes">
      <Command size={13} /> <span class="lbl">Rechercher</span> <span class="kbd">Ctrl K</span>
    </button>
    {#if rt}
      <button class="engine" onclick={() => (app.view = "models")} title={rt.message || rt.plan?.title || ""}>
        <span class="dot" class:ok={rt.state === "ready"} class:warn={rt.state === "degraded"} class:err={rt.state === "error"} class:busy={rt.state === "starting"}></span>
        <span class="name">{s2label}</span>
        {#if rt.state === "starting"}<span class="faint">demarrage…</span>{/if}
      </button>
    {/if}
    <button class="icon-btn" title="Theme clair / sombre" onclick={onTheme}>
      {#if theme === "dark"}<Sun size={15} />{:else}<Moon size={15} />{/if}
    </button>
    {#if isTauri && !mac}
      <div class="win">
        <button onclick={windowControls.minimize} aria-label="Reduire"><Minus size={14} /></button>
        <button onclick={windowControls.toggleMaximize} aria-label="Agrandir"><Square size={11} /></button>
        <button class="close" onclick={windowControls.close} aria-label="Fermer"><X size={15} /></button>
      </div>
    {/if}
  </div>
</header>

<style>
  .bar {
    height: var(--titlebar-h); flex: none; display: grid; grid-template-columns: minmax(max-content, 1fr) minmax(0, 1.2fr) minmax(max-content, 1fr); align-items: center;
    padding: 0 8px; border-bottom: 1px solid var(--line); background: color-mix(in srgb, var(--bg) 70%, transparent);
    -webkit-user-select: none; user-select: none; position: relative; z-index: 5;
  }
  .bar.mac { padding-left: 80px; }
  .left, .right { display: flex; align-items: center; gap: 6px; min-width: 0; }
  .right { justify-content: flex-end; }
  .brand { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-2); letter-spacing: -0.01em; padding-left: 2px; }
  .brand b { color: var(--text); font-weight: 650; }
  .demo { height: 18px; font-size: 10.5px; padding: 0 7px; }
  .center { display: flex; justify-content: center; min-width: 0; }
  .title { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; font-weight: 560; color: var(--text-2); padding: 4px 10px; border-radius: 8px; }
  .title:hover { background: var(--surface-2); color: var(--text); }
  .title-input { width: min(460px, 100%); height: 28px; text-align: center; background: var(--surface-2); border: 1px solid var(--line-3); border-radius: 8px; outline: none; font-size: 13px; }
  .palette { display: flex; align-items: center; gap: 8px; height: 28px; white-space: nowrap; flex: none; padding: 0 6px 0 10px; border-radius: 9px; color: var(--text-3); font-size: 12.5px; border: 1px solid var(--line); background: var(--surface); transition: border-color var(--t-fast), color var(--t-fast); }
  .palette:hover { color: var(--text-2); border-color: var(--line-3); }
  .engine { display: flex; align-items: center; gap: 8px; height: 28px; padding: 0 10px; border-radius: 9px; font-size: 12.5px; color: var(--text-2); white-space: nowrap; min-width: 0; }
  .engine .name { overflow: hidden; text-overflow: ellipsis; max-width: 200px; }
  .engine:hover { background: var(--surface-2); }
  .engine .name { color: var(--text); font-weight: 540; }
  .win { display: flex; margin-left: 4px; }
  .win button { width: 40px; height: var(--titlebar-h); display: grid; place-items: center; color: var(--text-2); transition: background var(--t-fast); }
  .win button:hover { background: var(--surface-3); color: var(--text); }
  .win .close:hover { background: #e5484d; color: white; }
  @media (max-width: 1100px) { .palette .lbl { display: none; } }
</style>
