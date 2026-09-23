<script lang="ts">
  import { Check, ChevronRight, CircleAlert, FileCode2, FilePen, FilePlus2, FileSearch, FolderTree, Globe, Hammer, NotebookPen, Search, SquareTerminal, Zap, Code2, Monitor } from "@lucide/svelte";
  import type { Block } from "../lib/types";
  import { app } from "../lib/store.svelte";
  import { api } from "../lib/api";
  import { highlight, langFromPath } from "../lib/markdown";
  import { ms, pct } from "../lib/format";
  import { parseUnifiedDiff } from "../lib/diff";
  import DiffView from "./DiffView.svelte";

  type Tool = Extract<Block, { type: "tool" }>;
  let { b }: { b: Tool } = $props();

  const META: Record<string, { verb: string; icon: any; open?: boolean }> = {
    write_file: { verb: "Ecrit", icon: FilePlus2, open: true },
    edit_file: { verb: "Modifie", icon: FilePen, open: true },
    read_file: { verb: "Lit", icon: FileCode2 },
    list_files: { verb: "Liste", icon: FolderTree },
    glob: { verb: "Trouve", icon: FileSearch },
    grep: { verb: "Cherche", icon: Search },
    run_command: { verb: "Execute", icon: SquareTerminal, open: true },
    python: { verb: "Python", icon: Code2, open: true },
    browse: { verb: "Navigue", icon: Globe, open: true },
    remember: { verb: "Memorise", icon: NotebookPen },
    create_tool: { verb: "Cree l'outil", icon: Hammer, open: true },
    desktop: { verb: "Pilote le bureau", icon: Monitor, open: true },
  };
  const meta = $derived(META[b.name] ?? (b.name.startsWith("judge_") ? { verb: "Juge", icon: Zap, open: true } : { verb: b.name, icon: Hammer }));
  const a = $derived(b.args as Record<string, any>);
  const target = $derived(
    a.path ?? a.command ?? a.pattern ?? a.goal ?? a.name ?? a.question ?? (b.name === "python" ? (a.code ?? "").split("\n")[0] : a.note) ?? "",
  );
  let open = $state<boolean | null>(null);
  const isOpen = $derived(open ?? (!!meta.open && b.status !== "running" && (b.name !== "run_command" || !!(b.result?.stdout || b.result?.stderr))));
  const stats = $derived(b.ui?.diff ? parseUnifiedDiff(b.ui.diff) : null);
  const dur = $derived(b.started && b.ended ? (b.ended - b.started) * 1000 : undefined);
  const r = $derived(b.result ?? {});
  const judge = $derived(b.name.startsWith("judge_"));

  function openFile(p: string) {
    app.inspectorFile = p;
    app.inspectorOpen = true;
  }
</script>

<div class="tool" class:error={b.status === "error"} class:running={b.status === "running"}>
  <button class="head" onclick={() => (open = !isOpen)}>
    <span class="ic" class:s1={judge}><meta.icon size={14} /></span>
    <span class="verb">{meta.verb}</span>
    <span class="target mono">{target}</span>
    <span class="right">
      {#if stats && (stats.added || stats.removed)}
        <span class="delta mono"><b class="add">+{stats.added}</b> <b class="del">−{stats.removed}</b></span>
      {/if}
      {#if r.blocked}<span class="chip warn">bloque</span>{/if}
      {#if dur}<span class="faint tabnum">{ms(dur)}</span>{/if}
      {#if b.status === "running"}<span class="spinner"></span>{:else if b.status === "error"}<CircleAlert size={14} class="err" />{:else}<Check size={14} class="ok" />{/if}
      <ChevronRight size={14} class="chev {isOpen ? 'open' : ''}" />
    </span>
  </button>

  {#if isOpen}
    <div class="body">
      {#if r.error && !r.blocked}
        <div class="errbox">{r.error}</div>
      {:else if r.blocked}
        <div class="note">{r.reason ?? "Action bloquee."}</div>
      {:else if (b.name === "write_file" || b.name === "edit_file") && b.ui?.diff}
        <DiffView diff={b.ui.diff} />
        <div class="foot">
          {#if /\.(html?|svg|pdf|png|jpe?g)$/i.test(a.path ?? "")}
            <button class="btn ghost sm" onclick={() => api("/api/open", { body: { path: a.path, session: app.session?.id } })}>Ouvrir</button>
          {/if}
          <button class="btn ghost sm" onclick={() => openFile(a.path)}>Voir le fichier</button>
        </div>
      {:else if b.name === "run_command" || b.name === "python"}
        <div class="term mono">
          {#if b.name === "run_command"}<div class="cmd"><span class="ps">$</span> {a.command}</div>{:else}<pre class="code">{@html highlight(a.code ?? "", "python")}</pre>{/if}
          {#if r.stdout}<pre class="out">{r.stdout}</pre>{/if}
          {#if r.stderr}<pre class="out stderr">{r.stderr}</pre>{/if}
          {#if r.returncode != null}<div class="rc">code de sortie <b class:bad={r.returncode !== 0}>{r.returncode}</b></div>{/if}
        </div>
      {:else if b.name === "read_file" && r.content != null}
        <pre class="code preview mono">{@html highlight(String(r.content).split("\n").slice(0, 60).join("\n"), langFromPath(a.path ?? ""))}</pre>
        <div class="foot"><button class="btn ghost sm" onclick={() => openFile(a.path)}>Ouvrir dans l'apercu</button></div>
      {:else if b.name === "glob" || b.name === "list_files"}
        <div class="files">
          {#each (r.files ?? r.entries ?? []).slice(0, 80) as f (f)}
            <button class="file mono" onclick={() => openFile(f)}>{f}</button>
          {:else}<span class="faint">Aucun fichier.</span>{/each}
        </div>
      {:else if b.name === "grep"}
        <pre class="out mono">{(r.matches ?? []).join("\n") || "Aucune correspondance."}</pre>
      {:else if judge}
        <div class="judge">
          {#if r.probabilities}
            {#each Object.entries(r.probabilities as Record<string, number>).sort((x, y) => y[1] - x[1]).slice(0, 8) as [k, v] (k)}
              <div class="meter"><span class="mono">{k}</span><div class="bar"><i style="transform: scaleX({v})"></i></div><b>{pct(v)}</b></div>
            {/each}
          {:else if r.noul != null}
            <div class="meter"><span>oui</span><div class="bar"><i style="transform: scaleX({r.noul})"></i></div><b>{pct(r.noul)}</b></div>
          {:else if r.ranked}
            {#each r.ranked.slice(0, 8) as c (c.candidate)}
              <div class="meter"><span class="mono">{c.candidate}</span><div class="bar"><i style="transform: scaleX({c.probability})"></i></div><b>{pct(c.probability)}</b></div>
            {/each}
          {/if}
        </div>
      {:else if b.name === "create_tool"}
        <pre class="code mono">{@html highlight(a.python_body ?? "", "python")}</pre>
      {:else if b.name === "desktop"}
        <div class="note">
          {r.window ?? ""} <span class="faint">{r.app ?? ""}</span> · {r.steps ?? 0} pas{r.fast_steps != null ? ` dont ${r.fast_steps} decides par le classifieur` : ""} · {r.status ?? ""}
        </div>
      {:else if b.name === "browse"}
        <div class="note">{r.title ?? ""} <span class="faint">{r.url ?? ""}</span> · {r.steps ?? 0} pas · {r.status ?? ""}</div>
      {:else}
        <pre class="out mono">{JSON.stringify(r, null, 2).slice(0, 4000)}</pre>
      {/if}
    </div>
  {/if}
</div>

<style>
  .tool { margin: 6px 0; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); overflow: hidden; animation: rise 260ms var(--ease) both; }
  .tool.running { border-color: color-mix(in srgb, var(--s2) 35%, var(--line)); }
  .tool.error { border-color: color-mix(in srgb, var(--err) 35%, var(--line)); }
  .head { width: 100%; display: flex; align-items: center; gap: 9px; padding: 8px 12px; text-align: left; min-width: 0; }
  .head:hover { background: var(--surface-2); }
  .ic { display: grid; place-items: center; width: 24px; height: 24px; border-radius: 7px; background: var(--surface-3); color: var(--text-2); flex: none; }
  .ic.s1 { color: var(--s1); background: var(--s1-soft); }
  .running .ic { color: var(--s2); background: var(--s2-soft); }
  .verb { font-size: 13px; font-weight: 600; flex: none; }
  .target { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text-2); font-size: 12px; }
  .right { display: flex; align-items: center; gap: 8px; flex: none; font-size: 11.5px; }
  .delta b { font-weight: 600; font-size: 11.5px; }
  .add { color: var(--ok); }
  .del { color: var(--err); }
  :global(.tool .ok) { color: var(--ok); }
  :global(.tool .err) { color: var(--err); }
  :global(.tool .chev) { color: var(--text-3); transition: transform var(--t-med) var(--ease); }
  :global(.tool .chev.open) { transform: rotate(90deg); }
  .running .spinner { color: var(--s2); }
  .body { border-top: 1px solid var(--line); }
  .foot { display: flex; justify-content: flex-end; gap: 4px; padding: 4px 6px; border-top: 1px solid var(--line); background: var(--surface); }
  .term { background: var(--bg-2); font-size: 12px; }
  .cmd { padding: 9px 14px; color: var(--text); border-bottom: 1px solid var(--line); white-space: pre-wrap; word-break: break-all; }
  .ps { color: var(--s1); margin-right: 4px; }
  .out { margin: 0; padding: 10px 14px; max-height: 300px; overflow: auto; white-space: pre-wrap; word-break: break-word; font-size: 12px; color: var(--text-2); background: var(--bg-2); }
  .stderr { color: color-mix(in srgb, var(--text-2) 60%, var(--err)); border-top: 1px solid var(--line); }
  .rc { padding: 6px 14px; font-size: 11px; color: var(--text-3); border-top: 1px solid var(--line); }
  .rc b { color: var(--ok); }
  .rc b.bad { color: var(--err); }
  .code { margin: 0; padding: 10px 14px; max-height: 360px; overflow: auto; font-size: 12px; line-height: 1.6; background: var(--bg-2); }
  .files { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 12px; }
  .file { font-size: 11.5px; padding: 3px 8px; border-radius: 6px; background: var(--surface-2); border: 1px solid var(--line); color: var(--text-2); }
  .file:hover { color: var(--text); border-color: var(--line-3); }
  .judge { padding: 10px 14px; display: flex; flex-direction: column; gap: 6px; }
  .meter { display: grid; grid-template-columns: minmax(80px, 200px) 1fr 44px; gap: 10px; align-items: center; font-size: 12px; color: var(--text-2); }
  .meter .bar i { background: var(--s1); }
  .meter b { text-align: right; font-weight: 560; color: var(--text); }
  .errbox { padding: 10px 14px; color: var(--err); font-size: 12.5px; background: var(--err-soft); }
  .note { padding: 10px 14px; font-size: 12.5px; color: var(--text-2); }
</style>
