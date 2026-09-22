<script lang="ts">
  import { onMount } from "svelte";
  import { app } from "./lib/store.svelte";
  import { onTauriEvent } from "./lib/api";
  import { commands } from "./lib/commands";
  import { speaker, startPtt, stopPtt } from "./lib/voice";
  import TitleBar from "./components/TitleBar.svelte";
  import Sidebar from "./components/Sidebar.svelte";
  import ChatView from "./components/ChatView.svelte";
  import ModelsView from "./components/ModelsView.svelte";
  import SettingsView from "./components/SettingsView.svelte";
  import StatusBar from "./components/StatusBar.svelte";
  import CommandPalette from "./components/CommandPalette.svelte";
  import Toasts from "./components/Toasts.svelte";
  import SetupWizard from "./components/SetupWizard.svelte";
  import VoiceOrb from "./components/VoiceOrb.svelte";
  import Inspector from "./components/Inspector.svelte";
  import CoreBoot from "./components/CoreBoot.svelte";

  let hidden = $state(false);
  let theme = $state<"dark" | "light">("dark");

  $effect(() => {
    document.documentElement.dataset.theme = theme;
  });

  const thinking = $derived(app.running);

  function run(id: string) {
    commands.find((c) => c.id === id)?.run();
  }

  function onKey(e: KeyboardEvent) {
    const mod = e.ctrlKey || e.metaKey;
    const typing = (e.target as HTMLElement)?.closest?.("input, textarea, [contenteditable]");
    if (mod && e.key.toLowerCase() === "k") {
      e.preventDefault();
      app.paletteOpen = !app.paletteOpen;
    } else if (mod && e.shiftKey && e.code === "Space") {
      e.preventDefault();
      if (!e.repeat) startPtt();
    } else if (mod && e.key.toLowerCase() === "n") {
      e.preventDefault();
      run("new_session");
    } else if (mod && e.key.toLowerCase() === "b") {
      e.preventDefault();
      run("files");
    } else if (mod && e.key === "\\") {
      e.preventDefault();
      app.sidebarOpen = !app.sidebarOpen;
    } else if (mod && e.key === ",") {
      e.preventDefault();
      run("open_settings");
    } else if (e.key === "Escape" && !app.paletteOpen && !app.wizardOpen) {
      if (app.voice.speaking) speaker.stop();
      else if (app.running && !typing) app.cancel();
    }
  }

  function onKeyUp(e: KeyboardEvent) {
    if (app.voice.recording && !app.voice.handsfree && (e.code === "Space" || e.key === "Shift" || e.key === "Control" || e.key === "Meta")) stopPtt();
  }

  onMount(() => {
    try {
      const t = localStorage.getItem("prophet.theme");
      if (t === "light" || t === "dark") theme = t;
    } catch {
      /* stockage indisponible : theme sombre */
    }
    app.init();
    app.onTurnEnd = (item) => {
      if (app.settings?.voice.speak_responses && item.response && item.stopped_by !== "cancelled") speaker.speak(item.response);
    };
    const vis = () => (hidden = document.hidden);
    document.addEventListener("visibilitychange", vis);
    onTauriEvent("shortcut://ptt", (p) => (p.state === "pressed" ? startPtt() : stopPtt()));
    return () => document.removeEventListener("visibilitychange", vis);
  });

  export function toggleTheme() {
    theme = theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem("prophet.theme", theme);
    } catch {
      /* ignore */
    }
  }
</script>

<svelte:window onkeydown={onKey} onkeyup={onKeyUp} />

<div class="shell" class:paused={hidden} class:reduce-motion={app.settings?.reduce_motion}>
  <div class="aura" class:thinking aria-hidden="true">
    <i class="a1"></i><i class="a2"></i>
  </div>
  <TitleBar onTheme={toggleTheme} {theme} />
  <div class="body">
    {#if app.sidebarOpen && app.core}
      <Sidebar />
    {/if}
    <main>
      {#if app.coreStatus !== "ready" || !app.core}
        <CoreBoot />
      {:else if app.view === "models"}
        <ModelsView />
      {:else if app.view === "settings"}
        <SettingsView onTheme={toggleTheme} {theme} />
      {:else}
        <ChatView />
      {/if}
    </main>
    {#if app.inspectorOpen && app.view === "chat" && app.core}
      <Inspector />
    {/if}
  </div>
  <StatusBar />
  <CommandPalette />
  <Toasts />
  <VoiceOrb />
  {#if app.wizardOpen && app.core}
    <SetupWizard />
  {/if}
</div>

<style>
  .shell { position: relative; height: 100%; display: flex; flex-direction: column; isolation: isolate; }
  .body { flex: 1; min-height: 0; display: flex; position: relative; }
  main { flex: 1; min-width: 0; display: flex; flex-direction: column; position: relative; }
  .aura { position: absolute; inset: 0; pointer-events: none; z-index: -1; overflow: hidden; }
  .aura i { position: absolute; border-radius: 50%; opacity: 0.55; transition: opacity 1.2s var(--ease), transform 1.2s var(--ease); }
  .a1 { width: 900px; height: 520px; left: 50%; top: -360px; transform: translateX(-60%); background: radial-gradient(closest-side, color-mix(in srgb, var(--s2) 22%, transparent), transparent); }
  .a2 { width: 700px; height: 420px; right: -220px; top: -250px; background: radial-gradient(closest-side, color-mix(in srgb, var(--s1) 12%, transparent), transparent); }
  .aura.thinking .a1 { opacity: 0.95; transform: translateX(-55%) scale(1.06); }
  .aura.thinking .a2 { opacity: 0.35; }
  :global(:root[data-theme="light"]) .aura i { opacity: 0.35; }
</style>
