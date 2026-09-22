<script lang="ts">
  import { ListChecks } from "@lucide/svelte";
  import type { UserItem } from "../lib/types";

  let { item }: { item: UserItem } = $props();
  const parts = $derived(item.text.split(/(@[\w./\\-]+)/g));
</script>

<div class="user rise">
  <div class="bubble">
    {#if item.plan_mode}<span class="chip s2 plan"><ListChecks size={11} /> plan</span>{/if}
    <p>{#each parts as p, i (i)}{#if p.startsWith("@")}<span class="mention">{p}</span>{:else}{p}{/if}{/each}</p>
  </div>
</div>

<style>
  .user { display: flex; justify-content: flex-end; padding: 8px 0 18px; }
  .bubble { max-width: min(78%, 680px); padding: 10px 16px; border-radius: 18px 18px 6px 18px; background: var(--surface-3); border: 1px solid var(--line); box-shadow: var(--shadow-1); }
  p { margin: 0; white-space: pre-wrap; font-size: 14.5px; line-height: 1.6; overflow-wrap: anywhere; }
  .plan { height: 19px; font-size: 10.5px; margin-bottom: 6px; }
  .mention { color: var(--s1); font-family: var(--mono); font-size: 0.9em; background: var(--s1-soft); padding: 0 4px; border-radius: 5px; }
</style>
