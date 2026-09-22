<script lang="ts">
  import { Brain, ChevronRight } from "@lucide/svelte";
  import { ms } from "../lib/format";

  let { text, live = false, duration }: { text: string; live?: boolean; duration?: number } = $props();
  let open = $state(false);
  const tail = $derived(text.trim().split("\n").slice(-3).join("\n").slice(-360));
  const tokens = $derived(Math.round(text.length / 3.6));
</script>

<div class="think" class:live>
  <button class="head" onclick={() => (open = !open)} aria-expanded={open}>
    <Brain size={14} />
    {#if live}
      <span class="shimmer">Bonsai reflechit…</span>
    {:else}
      <span>Reflexion{duration ? ` · ${ms(duration)}` : ""} · ~{tokens} tokens</span>
    {/if}
    <ChevronRight size={14} class="chev {open ? 'open' : ''}" />
  </button>
  {#if open}
    <div class="body fade-in">{text}</div>
  {:else if live && tail}
    <div class="peek">{tail}</div>
  {/if}
</div>

<style>
  .think { margin: 4px 0 12px; }
  .head { display: inline-flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--text-3); padding: 4px 8px 4px 6px; margin-left: -6px; border-radius: 8px; }
  .head:hover { color: var(--text-2); background: var(--surface-2); }
  .live .head { color: var(--s2); }
  :global(.think .chev) { transition: transform var(--t-med) var(--ease); }
  :global(.think .chev.open) { transform: rotate(90deg); }
  .body, .peek { font-size: 13px; line-height: 1.6; color: var(--text-3); white-space: pre-wrap; border-left: 2px solid color-mix(in srgb, var(--s2) 40%, transparent); padding: 4px 0 4px 14px; margin: 6px 0 0 6px; }
  .body { max-height: 420px; overflow-y: auto; }
  .peek { -webkit-mask-image: linear-gradient(to bottom, transparent, black 40%); mask-image: linear-gradient(to bottom, transparent, black 40%); max-height: 5.2em; overflow: hidden; }
</style>
