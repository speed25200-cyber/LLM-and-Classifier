<script lang="ts">
  import { Cpu, MessageSquare, Pencil, Plus, Search, Settings2, Trash2, Boxes } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import { ago, mibToGib } from "../lib/format";
  import type { SessionSummary } from "../lib/types";

  let q = $state("");
  let renaming = $state<string | null>(null);
  let draft = $state("");

  const groups = $derived.by(() => {
    const list = (app.core?.sessions ?? []).filter((s) => !q || s.title.toLowerCase().includes(q.toLowerCase()));
    const now = Date.now() / 1000;
    const out: { label: string; items: SessionSummary[] }[] = [
      { label: "Aujourd'hui", items: [] },
      { label: "7 derniers jours", items: [] },
      { label: "Plus ancien", items: [] },
    ];
    for (const s of list) out[now - s.updated < 86400 ? 0 : now - s.updated < 7 * 86400 ? 1 : 2].items.push(s);
    return out.filter((g) => g.items.length);
  });

  const g = $derived(app.core?.hardware.primary_gpu);
  const used = $derived(app.metrics?.vram_used_mib ?? g?.vram_used_mib ?? 0);
  const total = $derived(app.metrics?.vram_total_mib ?? g?.vram_total_mib ?? 1);

  async function commitRename(id: string) {
    if (draft.trim()) await app.renameSession(id, draft.trim());
    renaming = null;
  }
</script>

<aside class="side">
  <div class="top">
    <button class="btn new" onclick={() => app.newSession()}>
      <Plus size={16} /> Nouvelle session <span class="kbd">Ctrl N</span>
    </button>
    <label class="search">
      <Search size={14} />
      <input placeholder="Rechercher une session" bind:value={q} />
    </label>
  </div>

  <nav class="sessions">
    {#each groups as grp (grp.label)}
      <div class="group">{grp.label}</div>
      {#each grp.items as s (s.id)}
        <div class="row" class:active={app.session?.id === s.id && app.view === "chat"}>
          {#if renaming === s.id}
            <!-- svelte-ignore a11y_autofocus -->
            <input class="rename" bind:value={draft} autofocus onblur={() => commitRename(s.id)} onkeydown={(e) => e.key === "Enter" ? commitRename(s.id) : e.key === "Escape" && (renaming = null)} />
          {:else}
            <button class="open" onclick={() => app.openSession(s.id)}>
              {#if app.core?.running.includes(s.id)}<span class="spinner live"></span>{/if}
              <span class="t">{s.title}</span>
              <span class="when">{ago(s.updated)}</span>
            </button>
            <div class="acts">
              <button class="icon-btn" title="Renommer" onclick={() => ((renaming = s.id), (draft = s.title))}><Pencil size={13} /></button>
              <button class="icon-btn" title="Supprimer" onclick={() => app.deleteSession(s.id)}><Trash2 size={13} /></button>
            </div>
          {/if}
        </div>
      {/each}
    {:else}
      <p class="empty">Vos conversations apparaitront ici.</p>
    {/each}
  </nav>

  <div class="bottom">
    <div class="nav">
      <button class:on={app.view === "chat"} onclick={() => (app.view = "chat")}><MessageSquare size={15} /> Conversation</button>
      <button class:on={app.view === "models"} onclick={() => (app.view = "models")}><Boxes size={15} /> Modeles</button>
      <button class:on={app.view === "settings"} onclick={() => (app.view = "settings")}><Settings2 size={15} /> Reglages</button>
    </div>
    {#if g}
      <button class="gpu" onclick={() => (app.view = "models")}>
        <div class="gl"><Cpu size={14} /><span class="n">{g.name.replace("NVIDIA GeForce ", "")}</span>{#if g.is_blackwell}<span class="arch">Blackwell</span>{/if}</div>
        <div class="bar"><i style="transform: scaleX({Math.min(1, used / total)})"></i></div>
        <div class="gm tabnum"><span>VRAM {mibToGib(used)} / {mibToGib(total)}</span><span>{g.bandwidth_gbs} Go/s</span></div>
      </button>
    {/if}
  </div>
</aside>

<style>
  .side { width: 268px; flex: none; display: flex; flex-direction: column; border-right: 1px solid var(--line); background: color-mix(in srgb, var(--bg-2) 80%, transparent); animation: slideIn var(--t-med) var(--ease); }
  @keyframes slideIn { from { transform: translateX(-12px); opacity: 0; } }
  .top { padding: 12px 12px 8px; display: flex; flex-direction: column; gap: 8px; }
  .new { width: 100%; justify-content: flex-start; height: 38px; }
  .new .kbd { margin-left: auto; }
  .search { display: flex; align-items: center; gap: 8px; height: 32px; padding: 0 10px; border-radius: 9px; color: var(--text-3); background: var(--surface); border: 1px solid var(--line); }
  .search input { flex: 1; min-width: 0; background: none; border: 0; outline: none; font-size: 13px; }
  .sessions { flex: 1; overflow-y: auto; padding: 4px 8px 12px; }
  .group { font-size: 11px; font-weight: 600; color: var(--text-3); padding: 12px 8px 6px; letter-spacing: 0.02em; }
  .row { position: relative; display: flex; align-items: center; border-radius: 9px; transition: background var(--t-fast); }
  .row:hover, .row.active { background: var(--surface-2); }
  .row.active::before { content: ""; position: absolute; left: -8px; top: 9px; bottom: 9px; width: 3px; border-radius: 3px; background: var(--grad); }
  .open { flex: 1; min-width: 0; display: flex; align-items: center; gap: 8px; padding: 8px 10px; text-align: left; }
  .t { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; color: var(--text-2); }
  .row.active .t, .row:hover .t { color: var(--text); }
  .when { font-size: 11px; color: var(--text-4); flex: none; }
  .live { width: 11px; height: 11px; color: var(--s2); }
  .acts { position: absolute; right: 4px; display: none; background: var(--surface-2); border-radius: 8px; }
  .row:hover .acts { display: flex; }
  .row:hover .when { visibility: hidden; }
  .acts .icon-btn { width: 26px; height: 26px; }
  .rename { flex: 1; margin: 3px; height: 30px; padding: 0 8px; border-radius: 7px; background: var(--surface); border: 1px solid var(--line-3); outline: none; font-size: 13px; }
  .empty { color: var(--text-3); font-size: 12.5px; padding: 12px 10px; }
  .bottom { border-top: 1px solid var(--line); padding: 8px; display: flex; flex-direction: column; gap: 8px; }
  .nav { display: flex; flex-direction: column; gap: 1px; }
  .nav button { display: flex; align-items: center; gap: 10px; height: 32px; padding: 0 10px; border-radius: 8px; font-size: 13px; color: var(--text-2); transition: background var(--t-fast), color var(--t-fast); }
  .nav button:hover { background: var(--surface-2); color: var(--text); }
  .nav button.on { color: var(--text); background: var(--surface-2); }
  .gpu { text-align: left; padding: 10px 12px; border-radius: 12px; background: var(--surface); border: 1px solid var(--line); display: flex; flex-direction: column; gap: 7px; transition: border-color var(--t-fast); }
  .gpu:hover { border-color: var(--line-3); }
  .gl { display: flex; align-items: center; gap: 7px; font-size: 12.5px; color: var(--text-2); }
  .gl .n { color: var(--text); font-weight: 580; flex: 1; }
  .arch { font-size: 10.5px; padding: 1px 6px; border-radius: 5px; color: var(--s1); background: var(--s1-soft); }
  .gm { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-3); }
</style>
