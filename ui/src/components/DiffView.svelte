<script lang="ts">
  import { parseUnifiedDiff, simpleDiff, type DiffRow } from "../lib/diff";

  let { diff = "", oldText, newText, max = 400 }: { diff?: string; oldText?: string; newText?: string; max?: number } = $props();
  const rows = $derived<DiffRow[]>(oldText != null && newText != null ? simpleDiff(oldText, newText) : parseUnifiedDiff(diff).rows);
  let expanded = $state(false);
  const shown = $derived(expanded ? rows : rows.slice(0, max));
</script>

<div class="diff mono">
  {#each shown as r, i (i)}
    <div class="ln {r.kind}">
      {#if r.kind === "hunk"}
        <span class="hunk">{r.text}</span>
      {:else}
        <span class="no">{r.oldNo ?? ""}</span><span class="no">{r.newNo ?? ""}</span>
        <span class="sign">{r.kind === "add" ? "+" : r.kind === "del" ? "−" : " "}</span>
        <span class="code">{r.text || " "}</span>
      {/if}
    </div>
  {/each}
  {#if rows.length > shown.length}
    <button class="more" onclick={() => (expanded = true)}>Afficher les {rows.length - shown.length} lignes restantes</button>
  {/if}
</div>

<style>
  .diff { font-size: 12px; line-height: 1.62; overflow: auto; max-height: 440px; background: var(--bg-2); }
  .ln { display: grid; grid-template-columns: 38px 38px 16px 1fr; min-width: max-content; }
  .ln.add { background: var(--add-bg); }
  .ln.del { background: var(--del-bg); }
  .no { color: var(--text-4); text-align: right; padding-right: 8px; user-select: none; font-size: 11px; }
  .sign { color: var(--text-3); user-select: none; }
  .add .sign { color: var(--ok); }
  .del .sign { color: var(--err); }
  .code { white-space: pre; padding-right: 16px; color: var(--text); }
  .del .code { color: color-mix(in srgb, var(--text) 75%, var(--err)); }
  .ln.hunk { grid-template-columns: 1fr; }
  .hunk { color: var(--s2); padding: 2px 12px; background: var(--s2-soft); font-size: 11px; }
  .more { width: 100%; padding: 8px; font-size: 12px; color: var(--text-2); border-top: 1px solid var(--line); font-family: var(--font); }
  .more:hover { background: var(--surface-2); }
</style>
