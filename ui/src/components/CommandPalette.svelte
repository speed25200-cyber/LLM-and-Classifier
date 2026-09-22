<script lang="ts">
  import { CornerDownLeft, MessageSquare, Search } from "@lucide/svelte";
  import { tick } from "svelte";
  import { app } from "../lib/store.svelte";
  import { commands, fuzzy } from "../lib/commands";

  let q = $state("");
  let sel = $state(0);
  let input: HTMLInputElement | undefined = $state();

  interface Row { id: string; title: string; group: string; keys?: string; run: () => void }
  const rows = $derived.by<Row[]>(() => {
    const cmds: Row[] = commands.map((c) => ({ id: c.id, title: c.title, group: c.group, keys: c.keys ?? c.slash, run: () => c.run() }));
    const sess: Row[] = (app.core?.sessions ?? []).slice(0, 30).map((s) => ({ id: "s:" + s.id, title: s.title, group: "Sessions", run: () => app.openSession(s.id) }));
    const scored = [...cmds, ...sess].map((r) => ({ r, s: fuzzy(q, r.title + " " + r.group) })).filter((x) => x.s > 0);
    const best: Record<string, number> = {};
    for (const x of scored) best[x.r.group] = Math.max(best[x.r.group] ?? 0, x.s);
    // groupes ordonnes par leur meilleur resultat, puis resultats par score : un en-tete par groupe
    return scored
      .sort((a, b) => best[b.r.group] - best[a.r.group] || a.r.group.localeCompare(b.r.group) || b.s - a.s)
      .slice(0, 14)
      .map((x) => x.r);
  });

  $effect(() => {
    if (app.paletteOpen) {
      q = "";
      sel = 0;
      tick().then(() => input?.focus());
    }
  });
  $effect(() => {
    void q;
    sel = 0;
  });

  function go(r: Row) {
    app.paletteOpen = false;
    r.run();
  }
  function key(e: KeyboardEvent) {
    if (e.key === "ArrowDown") (e.preventDefault(), (sel = Math.min(rows.length - 1, sel + 1)));
    else if (e.key === "ArrowUp") (e.preventDefault(), (sel = Math.max(0, sel - 1)));
    else if (e.key === "Enter" && rows[sel]) (e.preventDefault(), go(rows[sel]));
    else if (e.key === "Escape") (e.preventDefault(), (app.paletteOpen = false));
  }
</script>

{#if app.paletteOpen}
  <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
  <div class="scrim fade-in" onclick={() => (app.paletteOpen = false)}>
    <div class="pal rise" onclick={(e) => e.stopPropagation()} role="dialog" tabindex="-1" aria-label="Palette de commandes">
      <label class="in"><Search size={16} /><input bind:this={input} bind:value={q} onkeydown={key} placeholder="Tapez une commande ou le nom d'une session…" /></label>
      <div class="list">
        {#each rows as r, i (r.id)}
          {#if i === 0 || rows[i - 1].group !== r.group}<div class="grp">{r.group}</div>{/if}
          <button class:on={i === sel} onmouseenter={() => (sel = i)} onclick={() => go(r)}>
            {#if r.group === "Sessions"}<MessageSquare size={14} />{/if}
            <span class="t">{r.title}</span>
            {#if r.keys}<span class="kbd">{r.keys}</span>{/if}
            {#if i === sel}<CornerDownLeft size={13} class="ret" />{/if}
          </button>
        {:else}
          <p class="faint none">Aucun resultat.</p>
        {/each}
      </div>
    </div>
  </div>
{/if}

<style>
  .scrim { position: fixed; inset: 0; z-index: 50; background: rgba(4, 5, 8, 0.55); display: flex; justify-content: center; padding-top: 14vh; }
  .pal { width: min(640px, 92vw); max-height: 60vh; display: flex; flex-direction: column; border-radius: 18px; background: var(--surface-2); box-shadow: var(--shadow-3); overflow: hidden; }
  .in { display: flex; align-items: center; gap: 12px; padding: 0 18px; height: 56px; border-bottom: 1px solid var(--line); color: var(--text-3); }
  .in input { flex: 1; background: none; border: 0; outline: none; font-size: 16px; color: var(--text); }
  .list { overflow-y: auto; padding: 6px; }
  .grp { font-size: 11px; font-weight: 600; color: var(--text-3); padding: 10px 12px 4px; letter-spacing: 0.03em; }
  .list button { width: 100%; display: flex; align-items: center; gap: 10px; height: 38px; padding: 0 12px; border-radius: 10px; font-size: 13.5px; color: var(--text-2); text-align: left; }
  .list button.on { background: var(--surface-4); color: var(--text); }
  .t { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  :global(.pal .ret) { color: var(--text-3); }
  .none { padding: 14px; font-size: 13px; }
</style>
