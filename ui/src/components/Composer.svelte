<script lang="ts">
  import { ArrowUp, Brain, ListChecks, Mic, ShieldCheck, Square, Zap, FileText, Gauge } from "@lucide/svelte";
  import { tick } from "svelte";
  import { app } from "../lib/store.svelte";
  import { api } from "../lib/api";
  import { fuzzy, matchSlash, type Command } from "../lib/commands";
  import { startPtt, stopPtt } from "../lib/voice";
  import type { PermissionMode } from "../lib/types";

  let ta: HTMLTextAreaElement | undefined = $state();
  let focused = $state(false);
  let sel = $state(0);
  let files = $state<string[]>([]);
  let filesFor = "";
  let mention = $state<{ start: number; q: string } | null>(null);
  let permOpen = $state(false);

  const slash = $derived(/^\/\S*$/.test(app.composerText) ? matchSlash(app.composerText) : []);
  const mentionHits = $derived(
    mention ? files.map((f) => [f, fuzzy(mention!.q, f)] as const).filter(([, s]) => s > 0).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([f]) => f) : [],
  );
  const menuLen = $derived(slash.length || mentionHits.length);
  const PERM: Record<PermissionMode, { label: string; desc: string }> = {
    smart: { label: "Smart", desc: "Le classifieur juge chaque action et ne demande que si elle est risquee" },
    ask: { label: "Demander", desc: "Confirmer chaque ecriture et chaque commande" },
    auto: { label: "Auto", desc: "Ne jamais demander (a vos risques)" },
  };
  const perm = $derived((app.settings?.permission_mode ?? "smart") as PermissionMode);
  const effort = $derived(app.effortOnce ?? app.settings?.effort ?? "auto");

  $effect(() => {
    void app.composerFocus;
    tick().then(() => ta?.focus());
  });

  $effect(() => {
    void app.composerText;
    autosize();
  });

  function autosize() {
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, window.innerHeight * 0.4) + "px";
  }

  async function loadFiles() {
    const sid = app.session?.id ?? "";
    if (filesFor === sid + app.filesVersion) return;
    filesFor = sid + app.filesVersion;
    try {
      files = (await api<{ entries: string[] }>(`/api/files${sid ? `?session=${sid}` : ""}`)).entries.filter((f) => !f.endsWith("/"));
    } catch {
      files = [];
    }
  }

  function updateMention() {
    if (!ta) return;
    const pos = ta.selectionStart;
    const m = /(^|\s)@([\w./\\-]*)$/.exec(app.composerText.slice(0, pos));
    mention = m ? { start: pos - m[2].length - 1, q: m[2] } : null;
    if (mention) {
      sel = 0;
      loadFiles();
    }
  }

  function applyMention(f: string) {
    if (!mention || !ta) return;
    const end = ta.selectionStart;
    app.composerText = app.composerText.slice(0, mention.start) + "@" + f + " " + app.composerText.slice(end);
    mention = null;
    tick().then(() => ta?.focus());
  }

  async function runSlash(c: Command) {
    app.composerText = "";
    await c.run();
  }

  async function submit() {
    if (app.running) return;
    const t = app.composerText;
    if (!t.trim()) return;
    app.composerText = "";
    await app.send(t);
  }

  function onKey(e: KeyboardEvent) {
    if (menuLen) {
      if (e.key === "ArrowDown") return e.preventDefault(), (sel = (sel + 1) % menuLen);
      if (e.key === "ArrowUp") return e.preventDefault(), (sel = (sel - 1 + menuLen) % menuLen);
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        if (slash.length) runSlash(slash[sel]);
        else applyMention(mentionHits[sel]);
        return;
      }
      if (e.key === "Escape") return e.preventDefault(), (mention = null), slash.length && (app.composerText = "");
    }
    if (e.key === "Tab" && e.shiftKey) {
      e.preventDefault();
      app.planMode = !app.planMode;
      return;
    }
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      submit();
    } else if (e.key === "Escape" && app.running && !app.composerText) {
      app.cancel();
    }
  }

  // micro : maintenir = push-to-talk ; clic court = bascule
  let pressedAt = 0;
  function micDown(e: PointerEvent) {
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    pressedAt = performance.now();
    if (app.voice.recording) return;
    startPtt();
  }
  function micUp() {
    const held = performance.now() - pressedAt;
    if (held > 280 || !app.voice.recording) stopPtt();
  }

  function cycleEffort() {
    const order = ["auto", "fast", "deep"] as const;
    app.effortOnce = order[(order.indexOf(effort as (typeof order)[number]) + 1) % 3];
  }
</script>

<div class="dock">
  <div class="composer" class:focused class:plan={app.planMode}>
    {#if menuLen}
      <div class="menu rise" role="listbox">
        {#if slash.length}
          {#each slash as c, i (c.id)}
            <button class:on={i === sel} onmouseenter={() => (sel = i)} onclick={() => runSlash(c)} role="option" aria-selected={i === sel}>
              <span class="mono cmd">{c.slash}</span><span class="d">{c.title}</span>{#if c.keys}<span class="kbd">{c.keys}</span>{/if}
            </button>
          {/each}
        {:else}
          {#each mentionHits as f, i (f)}
            <button class:on={i === sel} onmouseenter={() => (sel = i)} onclick={() => applyMention(f)} role="option" aria-selected={i === sel}>
              <FileText size={13} /><span class="mono">{f}</span>
            </button>
          {/each}
        {/if}
      </div>
    {/if}

    <textarea
      bind:this={ta}
      bind:value={app.composerText}
      rows="1"
      placeholder={app.planMode ? "Decrivez l'objectif : Prophet explore puis propose un plan, sans rien modifier…" : "Demandez a Prophet…   / commandes · @ fichiers · Maj+Tab mode plan"}
      onkeydown={onKey}
      oninput={updateMention}
      onclick={updateMention}
      onfocus={() => (focused = true)}
      onblur={() => ((focused = false), setTimeout(() => (mention = null), 150))}
    ></textarea>

    <div class="tools">
      <div class="left">
        <div class="perm">
          <button class="pill" onclick={() => (permOpen = !permOpen)} title={PERM[perm].desc}><ShieldCheck size={13} /> {PERM[perm].label}</button>
          {#if permOpen}
            <div class="pop rise">
              {#each Object.entries(PERM) as [k, v] (k)}
                <button class:on={k === perm} onclick={() => (app.updateSettings({ permission_mode: k }), (permOpen = false))}>
                  <b>{v.label}</b><span>{v.desc}</span>
                </button>
              {/each}
            </div>
          {/if}
        </div>
        <button class="pill" class:active={app.planMode} onclick={() => (app.planMode = !app.planMode)} title="Mode plan : lecture seule (Maj+Tab)"><ListChecks size={13} /> Plan</button>
        <button class="pill" class:active={effort !== "auto"} onclick={cycleEffort} title="Effort de reflexion du prochain message (auto = choisi par le classifieur)">
          {#if effort === "deep"}<Brain size={13} /> Profond{:else if effort === "fast"}<Zap size={13} /> Rapide{:else}<Gauge size={13} /> Auto{/if}
        </button>
      </div>
      <div class="right">
        <button class="mic" class:rec={app.voice.recording && !app.voice.handsfree} class:busy={app.voice.busy} onpointerdown={micDown} onpointerup={micUp} title="Maintenir pour parler (Ctrl+Maj+Espace)">
          {#if app.voice.busy}<span class="spinner"></span>{:else}<Mic size={16} />{/if}
          <i style="transform: scale({1 + app.voice.level * 0.9})"></i>
        </button>
        {#if app.running}
          <button class="send stop" onclick={() => app.cancel()} title="Arreter (Echap)"><Square size={13} fill="currentColor" /></button>
        {:else}
          <button class="send" disabled={!app.composerText.trim()} onclick={submit} title="Envoyer (Entree)"><ArrowUp size={17} /></button>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .dock { padding: 8px 24px 16px; flex: none; }
  .composer {
    position: relative; max-width: 812px; margin: 0 auto; border-radius: 20px; background: var(--surface);
    border: 1px solid var(--line-2); box-shadow: var(--shadow-2); transition: border-color var(--t-med), box-shadow var(--t-med);
  }
  .composer.focused { border-color: var(--line-3); box-shadow: var(--shadow-2), 0 0 0 4px color-mix(in srgb, var(--s2) 9%, transparent); }
  .composer.plan { border-color: color-mix(in srgb, var(--s2) 45%, transparent); }
  textarea { display: block; width: 100%; resize: none; border: 0; outline: none; background: transparent; padding: 15px 18px 6px; font-size: 14.5px; line-height: 1.55; max-height: 40vh; }
  textarea::placeholder { color: var(--text-4); }
  .tools { display: flex; align-items: center; justify-content: space-between; padding: 6px 8px 8px 10px; gap: 8px; }
  .left, .right { display: flex; align-items: center; gap: 4px; }
  .pill { display: inline-flex; align-items: center; gap: 6px; height: 28px; padding: 0 10px; border-radius: 9px; font-size: 12px; font-weight: 560; color: var(--text-3); transition: background var(--t-fast), color var(--t-fast); }
  .pill:hover { background: var(--surface-3); color: var(--text); }
  .pill.active { color: var(--s2); background: var(--s2-soft); }
  .perm { position: relative; }
  .pop { position: absolute; bottom: 36px; left: 0; width: 300px; padding: 6px; border-radius: 14px; background: var(--surface-2); border: 1px solid var(--line-2); box-shadow: var(--shadow-3); z-index: 10; display: flex; flex-direction: column; gap: 2px; }
  .pop button { display: flex; flex-direction: column; align-items: flex-start; gap: 2px; padding: 8px 10px; border-radius: 9px; text-align: left; }
  .pop button:hover, .pop button.on { background: var(--surface-4); }
  .pop b { font-size: 13px; font-weight: 600; }
  .pop span { font-size: 11.5px; color: var(--text-3); }
  .mic { position: relative; width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center; color: var(--text-2); isolation: isolate; transition: color var(--t-fast), background var(--t-fast); touch-action: none; }
  .mic:hover { background: var(--surface-3); color: var(--text); }
  .mic i { position: absolute; inset: 4px; border-radius: 50%; background: color-mix(in srgb, var(--s1) 30%, transparent); z-index: -1; opacity: 0; transition: transform 80ms linear, opacity var(--t-fast); }
  .mic.rec { color: #06110f; background: var(--s1); }
  .mic.rec i { opacity: 1; }
  .mic.busy { color: var(--s1); }
  .send { width: 34px; height: 34px; border-radius: 11px; display: grid; place-items: center; background: var(--text); color: var(--bg); transition: transform var(--t-fast) var(--ease), opacity var(--t-fast), background var(--t-fast); }
  .send:hover { transform: translateY(-1px); }
  .send:disabled { opacity: 0.25; transform: none; cursor: default; }
  .send.stop { background: var(--surface-4); color: var(--text); }
  .menu { position: absolute; bottom: calc(100% + 8px); left: 0; right: 0; padding: 6px; border-radius: 16px; background: var(--surface-2); border: 1px solid var(--line-2); box-shadow: var(--shadow-3); z-index: 10; max-height: 320px; overflow-y: auto; }
  .menu button { width: 100%; display: flex; align-items: center; gap: 10px; padding: 8px 10px; border-radius: 10px; text-align: left; font-size: 13px; color: var(--text-2); }
  .menu button.on { background: var(--surface-4); color: var(--text); }
  .menu .cmd { color: var(--s1); min-width: 110px; }
  .menu .d { flex: 1; }
  @media (max-width: 720px) { .dock { padding: 8px 10px 12px; } }
</style>
