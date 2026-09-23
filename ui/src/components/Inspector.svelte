<script lang="ts">
  import { ChevronRight, ExternalLink, File, Folder, RefreshCw, ShieldCheck, X } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import { alwaysAllow, api } from "../lib/api";
  import { highlight, langFromPath } from "../lib/markdown";
  import { grantLabel } from "./PermissionCard.svelte";

  interface Node { name: string; path: string; dir: boolean; children: Node[] }
  let entries = $state<string[]>([]);
  let root = $state("");
  let content = $state<string | null>(null);
  let collapsed = $state<Record<string, boolean>>({});
  let loading = $state(false);

  async function load() {
    loading = true;
    try {
      const r = await api<{ root: string; entries: string[] }>(`/api/files${app.session ? `?session=${app.session.id}` : ""}`);
      entries = r.entries;
      root = r.root;
    } catch {
      entries = [];
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    void app.filesVersion;
    void app.session?.id;
    load();
  });

  $effect(() => {
    const f = app.inspectorFile;
    content = null;
    if (!f) return;
    api<{ content: string }>(`/api/file?path=${encodeURIComponent(f)}${app.session ? `&session=${app.session.id}` : ""}`)
      .then((r) => (content = r.content))
      .catch((e) => (content = `// ${e}`));
  });

  const tree = $derived.by(() => {
    const top: Node = { name: "", path: "", dir: true, children: [] };
    for (const e of entries) {
      const dir = e.endsWith("/");
      const parts = e.replace(/\/$/, "").split("/");
      let cur = top;
      parts.forEach((p, i) => {
        const path = parts.slice(0, i + 1).join("/");
        let n = cur.children.find((c) => c.name === p);
        if (!n) cur.children.push((n = { name: p, path, dir: dir || i < parts.length - 1, children: [] }));
        cur = n;
      });
    }
    const sort = (n: Node) => (n.children.sort((a, b) => Number(b.dir) - Number(a.dir) || a.name.localeCompare(b.name)), n.children.forEach(sort));
    sort(top);
    return top.children;
  });

  const lines = $derived(content != null ? highlight(content, langFromPath(app.inspectorFile ?? "")).split("\n") : []);

  // autorisations memorisees (« toujours pour cet outil ») de la session : visibles et revocables
  let grants = $state<string[]>([]);
  const decided = $derived(app.lastAssistant?.blocks.filter((b) => b.type === "permission" && b.decision).length ?? 0);

  async function loadGrants() {
    const sid = app.session?.id;
    if (!sid) return void (grants = []);
    try {
      grants = (await alwaysAllow.list(sid)).always_allow;
    } catch {
      grants = [];
    }
  }

  async function revoke(grant?: string) {
    const s = app.session;
    if (!s) return;
    try {
      grants = (await alwaysAllow.revoke(s.id, grant)).always_allow;
      s.always_allow = grants;
    } catch (e) {
      app.toast("error", "Revocation impossible", String(e));
    }
  }

  $effect(() => {
    void app.session?.id;
    void decided;
    void app.running;
    loadGrants();
  });
</script>

{#snippet branch(nodes: Node[], depth: number)}
  {#each nodes as n (n.path)}
    <button class="node" class:sel={app.inspectorFile === n.path} style="padding-left:{10 + depth * 14}px"
      onclick={() => (n.dir ? (collapsed[n.path] = !collapsed[n.path]) : (app.inspectorFile = n.path))}>
      {#if n.dir}<ChevronRight size={12} class="chev {collapsed[n.path] ? '' : 'open'}" /><Folder size={13} />{:else}<span class="sp"></span><File size={13} />{/if}
      <span class="nm">{n.name}</span>
    </button>
    {#if n.dir && !collapsed[n.path]}{@render branch(n.children, depth + 1)}{/if}
  {/each}
{/snippet}

<aside class="insp">
  <div class="head">
    <span class="panel-title">Espace de travail</span>
    <div>
      <button class="icon-btn" title="Actualiser" onclick={load}><RefreshCw size={13} class={loading ? "spin" : ""} /></button>
      <button class="icon-btn" title="Fermer" onclick={() => (app.inspectorOpen = false)}><X size={14} /></button>
    </div>
  </div>
  <div class="tree">
    {#if tree.length}{@render branch(tree, 0)}{:else}<p class="faint empty">Aucun fichier pour l'instant.</p>{/if}
  </div>
  {#if app.inspectorFile}
    <div class="preview">
      <div class="ph">
        <span class="mono">{app.inspectorFile}</span>
        <button class="icon-btn" title="Ouvrir avec l'application par defaut" onclick={() => api("/api/open", { body: { path: app.inspectorFile, session: app.session?.id } })}><ExternalLink size={13} /></button>
        <button class="icon-btn" title="Fermer l'apercu" onclick={() => (app.inspectorFile = null)}><X size={13} /></button>
      </div>
      <div class="code mono">
        {#if content == null}<div class="faint pad">Chargement…</div>{:else}
          {#each lines as l, i (i)}<div class="ln"><span class="no">{i + 1}</span><span class="c">{@html l || " "}</span></div>{/each}
        {/if}
      </div>
    </div>
  {/if}
  {#if grants.length}
    <div class="grants">
      <div class="gh">
        <span class="panel-title"><ShieldCheck size={12} /> Autorisations memorisees</span>
        {#if grants.length > 1}<button class="btn ghost sm" onclick={() => revoke()}>Tout revoquer</button>{/if}
      </div>
      {#each grants as g (g)}
        <div class="grant">
          <span class="gl" title={g}>{grantLabel(g)}</span>
          <button class="icon-btn" title="Revoquer : redemander a chaque fois" aria-label="Revoquer {grantLabel(g)}" onclick={() => revoke(g)}><X size={13} /></button>
        </div>
      {/each}
    </div>
  {/if}
  <div class="foot faint mono" title={root}>{root}</div>
</aside>

<style>
  .insp { width: 380px; flex: none; display: flex; flex-direction: column; border-left: 1px solid var(--line); background: color-mix(in srgb, var(--bg-2) 85%, transparent); animation: slideR var(--t-med) var(--ease); min-height: 0; }
  @keyframes slideR { from { transform: translateX(12px); opacity: 0; } }
  .head { display: flex; align-items: center; justify-content: space-between; padding: 8px 8px 6px 14px; }
  .head > div { display: flex; }
  .tree { flex: 1 1 40%; overflow-y: auto; padding: 2px 6px 8px; min-height: 80px; }
  .node { width: 100%; display: flex; align-items: center; gap: 6px; height: 27px; border-radius: 7px; font-size: 12.5px; color: var(--text-2); text-align: left; }
  .node:hover { background: var(--surface-2); color: var(--text); }
  .node.sel { background: var(--s2-soft); color: var(--text); }
  .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sp { width: 12px; flex: none; }
  :global(.insp .chev) { transition: transform var(--t-fast) var(--ease); flex: none; }
  :global(.insp .chev.open) { transform: rotate(90deg); }
  :global(.insp .spin) { animation: spin 0.8s linear infinite; }
  .empty { font-size: 12.5px; padding: 8px; }
  .preview { flex: 1 1 60%; display: flex; flex-direction: column; min-height: 0; border-top: 1px solid var(--line); }
  .ph { display: flex; align-items: center; gap: 4px; padding: 6px 6px 6px 14px; border-bottom: 1px solid var(--line); }
  .ph .mono { flex: 1; font-size: 11.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-2); }
  .code { flex: 1; overflow: auto; font-size: 11.8px; line-height: 1.6; background: var(--bg-2); padding: 6px 0; }
  .ln { display: grid; grid-template-columns: 42px 1fr; min-width: max-content; }
  .no { text-align: right; padding-right: 12px; color: var(--text-4); user-select: none; }
  .c { white-space: pre; padding-right: 16px; }
  .pad { padding: 10px 14px; }
  .grants { border-top: 1px solid var(--line); padding: 6px 6px 6px 14px; max-height: 160px; overflow-y: auto; }
  .gh { display: flex; align-items: center; justify-content: space-between; min-height: 26px; }
  .gh .panel-title { display: inline-flex; align-items: center; gap: 6px; }
  .grant { display: flex; align-items: center; gap: 6px; height: 27px; font-size: 12px; color: var(--text-2); }
  .gl { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .grant .icon-btn { width: 24px; height: 24px; }
  .foot { font-size: 10.5px; padding: 6px 14px; border-top: 1px solid var(--line); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
