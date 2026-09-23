<script lang="ts">
  import { Brain, Check, Cpu, Download, Gauge, Mic, Play, Power, RotateCw, Server, Target, Trash2, TriangleAlert, Upload, X, Zap, ScrollText } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import { api } from "../lib/api";
  import { bytes, ctxLabel, eta, gb, mibToGib, num } from "../lib/format";
  import type { DownloadJob, Priority } from "../lib/types";
  import VramBar from "./VramBar.svelte";

  const c = $derived(app.core!);
  const hw = $derived(c.hardware);
  const g = $derived(hw.primary_gpu);
  const plan = $derived(c.plan);
  const running = $derived(c.runtime.plan);
  const replanNeeded = $derived(
    !!running &&
      (running.s2.model_id !== plan.s2.model_id || running.s2.ctx !== plan.s2.ctx || running.s1?.device !== plan.s1?.device || running.s1?.model_id !== plan.s1?.model_id) &&
      running.rung === 0,
  );
  // un modele impose (catalogue ou GGUF importe) que le plan n'applique pas ne doit jamais passer en silence
  const unapplied = $derived(
    [
      { what: "Cerveau", want: c.settings.s2_model, got: plan.s2.model_id },
      { what: "Classifieur", want: c.settings.s1_model, got: plan.s1?.model_id ?? "" },
    ].filter((o) => o.want && o.want !== "auto" && o.want !== o.got),
  );
  // plan CPU qui ne tient pas en RAM (poids + KV > 85 % de la RAM) : jamais lance sans un accord explicite
  const ramShort = $derived(plan.backend === "cpu" && plan.rung === 0 && !plan.fits);
  const RAM_TITLE = "RAM insuffisante pour ce plan : voir l'avertissement de la configuration";
  const s1Installed = $derived(c.catalog.models.filter((m) => m.role === "s1" && c.installed.models[m.id]?.installed));
  const missingCustom = $derived(Object.entries(c.installed.models).filter(([, v]) => v.custom && !v.installed));
  let logs = $state<{ name: string; lines: string[] } | null>(null);
  let importPath = $state("");
  let importRole = $state<"s1" | "s2">("s1");
  let calPath = $state("");
  let calModel = $state("");
  const calTarget = $derived(calModel || app.activeS1 || s1Installed[0]?.id || "");

  const PRIO: { k: Priority; label: string; desc: string }[] = [
    { k: "equilibre", label: "Equilibre", desc: "Bonsai 2 entier sur GPU, contexte confortable" },
    { k: "contexte", label: "Contexte", desc: "Le plus long contexte possible (gros projets)" },
    { k: "vitesse", label: "Vitesse", desc: "Modele 1-bit plus rapide, classifieur sur GPU" },
  ];

  function jobs(group: string): DownloadJob[] {
    return app.downloadsFor(group).filter((j) => j.status !== "done" && j.status !== "cancelled");
  }
  function progress(js: DownloadJob[]) {
    const size = js.reduce((a, j) => a + (j.size || 0), 0);
    const done = js.reduce((a, j) => a + j.done_bytes, 0);
    const speed = js.reduce((a, j) => a + (j.status === "running" ? j.speed_bps : 0), 0);
    return { p: size ? done / size : null, done, size, speed, eta: speed ? (size - done) / speed : null, status: js[0]?.status, error: js.find((j) => j.error)?.error };
  }
  async function showLogs(name: string) {
    const r = await api<{ lines: string[] }>(`/api/runtime/logs/${name}`);
    logs = { name, lines: r.lines };
  }
  async function doImport() {
    try {
      const r = await api<{ id: string }>("/api/import", { body: { path: importPath, role: importRole } });
      app.toast("ok", "Modele importe", `${r.id} · choisi pour le prochain demarrage${importRole === "s1" ? " (pensez a le calibrer)" : ""}`, 7000);
      importPath = "";
      await app.updateSettings(importRole === "s1" ? { s1_model: r.id } : { s2_model: r.id });
    } catch (e) {
      app.toast("error", "Import impossible", String(e));
    }
  }
  async function doImportCalibration() {
    if (await app.importCalibration(calPath.trim(), calTarget)) calPath = "";
  }
  /** Retire un modele ; s'il etait choisi dans les reglages, le choix revient a l'automatique (sinon avertissement permanent). */
  async function removeModel(id: string) {
    const s = c.settings;
    try {
      await app.removeInstalled(id);
    } catch (e) {
      app.toast("error", "Retrait impossible", String(e));
      return;
    }
    const patch: Record<string, string> = {};
    if (s.s2_model === id) patch.s2_model = "auto";
    if (s.s1_model === id) patch.s1_model = "auto";
    if (Object.keys(patch).length) await app.updateSettings(patch);
  }
  const SERVERS = [
    { key: "s2" as const, label: "System Two · Bonsai", Icon: Brain },
    { key: "s1" as const, label: "System One · classifieur", Icon: Zap },
  ];
  const STATE_FR: Record<string, string> = { stopped: "arrete", starting: "demarrage", loading: "chargement", ready: "pret", crashed: "arret brutal", oom: "memoire insuffisante" };
</script>

{#snippet dlrow(id: string)}
  {@const js = jobs(id)}
  {#if js.length}
    {@const p = progress(js)}
    <div class="dl">
      <div class="bar {p.p == null ? 'indeterminate' : ''}"><i style="transform: scaleX({p.p ?? 0})"></i></div>
      <div class="dlm tabnum">
        <span>{p.status === "verifying" ? "Verification SHA-256…" : p.status === "extracting" ? "Extraction…" : p.size ? `${bytes(p.done)} / ${bytes(p.size)}` : bytes(p.done)}</span>
        <span>{p.speed ? `${bytes(p.speed)}/s · ${eta(p.eta)}` : p.error ?? ""}</span>
        <button class="icon-btn" title="Annuler" onclick={() => app.cancelDownloads(id)}><X size={13} /></button>
      </div>
    </div>
  {/if}
{/snippet}

<div class="page">
  <div class="grid top">
    <section class="card hw rise">
      <div class="panel-title">Materiel</div>
      {#if g}
        <div class="gpu">
          <div class="gicon"><Cpu size={22} /></div>
          <div>
            <h2>{g.name.replace("NVIDIA GeForce ", "")}</h2>
            <p class="faint">{mibToGib(g.vram_total_mib)} · {g.bandwidth_gbs ? `${g.bandwidth_gbs} Go/s` : ""} · pilote {g.driver || "?"} · CUDA {g.cuda_version || "?"}</p>
          </div>
        </div>
        <div class="badges">
          {#if g.is_blackwell}<span class="chip s1">Blackwell · sm_{g.compute_cap.replace(".", "")}</span>{:else if g.arch}<span class="chip">{g.arch} · sm_{g.compute_cap.replace(".", "")}</span>{/if}
          {#if g.display_active}<span class="chip warn">ecran branche sur ce GPU</span>{/if}
          <span class="chip">{hw.cpu_cores} coeurs · {num(hw.ram_total_gib)} Gio RAM</span>
          <span class="chip">{num(c.installed.free_disk_gb, 1)} Go libres</span>
        </div>
      {:else}
        <div class="gpu"><div class="gicon"><Cpu size={22} /></div><div><h2>CPU seul</h2><p class="faint">{hw.cpu} · {num(hw.ram_total_gib)} Gio RAM</p></div></div>
      {/if}
      {#each hw.warnings as w}<div class="warnbox"><TriangleAlert size={14} /> {w}</div>{/each}
    </section>

    <section class="card plan rise">
      <div class="ph">
        <div class="panel-title">Configuration optimale</div>
        <div class="seg">
          {#each PRIO as p (p.k)}
            <button class:on={c.settings.priority === p.k} title={p.desc} onclick={() => app.updateSettings({ priority: p.k })}>{p.label}</button>
          {/each}
        </div>
      </div>
      <h3>{plan.title}</h3>
      <VramBar {plan} />
      <div class="kpis tabnum">
        <div><span class="faint">Generation</span><b class="s2c"><Brain size={14} /> {plan.expected.s2.tok_s ? `${plan.expected.s2.tok_s[0]}-${plan.expected.s2.tok_s[1]} tok/s` : "a mesurer"}</b></div>
        <div><span class="faint">Decision</span><b class="s1c"><Zap size={14} /> {plan.expected.s1_ms[0]}-{plan.expected.s1_ms[1]} ms</b></div>
        <div><span class="faint">Contexte</span><b>{ctxLabel(plan.s2.ctx)} tokens</b></div>
        <div><span class="faint">Classifieur</span><b>{plan.s1?.device === "gpu" ? "GPU" : "CPU"}</b></div>
      </div>
      {#each unapplied as o (o.what)}
        <div class="warnbox">
          <TriangleAlert size={14} />
          <span class="wtxt">{o.what} choisi « {o.want} » non applique : le plan utilise {o.got || "aucun modele"} (voir les notes).</span>
          <button class="btn sm ghost" onclick={() => app.updateSettings(o.what === "Cerveau" ? { s2_model: "auto" } : { s1_model: "auto" })}>Revenir au choix automatique</button>
        </div>
      {/each}
      {#if ramShort}
        <div class="warnbox ram">
          <TriangleAlert size={14} />
          <span class="wtxt">
            RAM insuffisante : ce plan demande ~{mibToGib(plan.budget.ram_needed_mib ?? 0)} pour {mibToGib(plan.budget.ram_total_mib ?? 0)} de RAM
            (limite 85 %). Chargement tres lent (disque) ou echec : choisissez un modele plus petit.
          </span>
          <button class="btn sm danger" onclick={() => app.startRuntime()}>Lancer quand meme</button>
        </div>
      {/if}
      <ul class="notes">{#each plan.notes as n}<li>{n}</li>{/each}</ul>
      {#if replanNeeded && !ramShort}
        <button class="btn accent" onclick={() => app.startRuntime()}><RotateCw size={15} /> Appliquer et redemarrer</button>
      {/if}
    </section>
  </div>

  <section class="card runtime rise">
    <div class="ph">
      <div class="panel-title">Moteur</div>
      <div class="acts">
        {#if c.runtime.state === "ready" || c.runtime.state === "degraded"}
          <button class="btn sm" onclick={() => app.runBench()} disabled={app.benchRunning}>{#if app.benchRunning}<span class="spinner"></span>{:else}<Gauge size={14} />{/if} Mesurer</button>
          <button class="btn sm" onclick={() => app.startRuntime()} disabled={ramShort} title={ramShort ? RAM_TITLE : undefined}><RotateCw size={14} /> Redemarrer</button>
          <button class="btn sm ghost" onclick={() => app.stopRuntime()}><Power size={14} /> Arreter</button>
        {:else}
          <button class="btn sm accent" onclick={() => app.startRuntime()} disabled={c.runtime.state === "starting" || ramShort} title={ramShort ? RAM_TITLE : undefined}>
            <Play size={14} /> Demarrer
          </button>
        {/if}
      </div>
    </div>
    {#if c.runtime.message}<p class="msg {c.runtime.state === 'error' ? 'err' : 'faint'}">{c.runtime.message}</p>{/if}
    <div class="servers">
      {#each SERVERS as { key, label, Icon } (key)}
        {@const s = c.runtime.servers[key]}
        <div class="srv">
          <span class="sic {key}"><Icon size={16} /></span>
          <div class="st">
            <b>{label}</b>
            <span class="faint tabnum">
              <span class="dot {s.state === 'ready' ? 'ok' : s.state === 'crashed' || s.state === 'oom' ? 'err' : s.state === 'stopped' ? '' : 'busy'}"></span>
              {key === "s1" && c.runtime.mono ? "mode mono (Bonsai decide aussi)" : STATE_FR[s.state] ?? s.state}{s.port ? ` · :${s.port}` : ""}{s.load_s ? ` · charge en ${num(s.load_s, 1)} s` : ""}
            </span>
          </div>
          <button class="icon-btn" title="Journal" onclick={() => showLogs(key)}><ScrollText size={14} /></button>
        </div>
      {/each}
    </div>
    {#if c.bench}
      <div class="bench tabnum">
        <span class="chip s2"><Brain size={12} /> {num(c.bench.s2_tok_s, 1)} tok/s</span>
        <span class="chip">prefill {num(c.bench.s2_prefill_tok_s)} tok/s</span>
        <span class="chip s1" title={c.bench.s1_series?.s1_p50_ms ?? ""}><Zap size={12} /> decision p50 {num(c.bench.s1_p50_ms, 1)} ms · p95 {num(c.bench.s1_p95_ms, 1)} ms</span>
        {#if c.bench.s1_cold_p50_ms != null}
          <span class="chip s1" title={c.bench.s1_series?.s1_cold ?? ""}>
            etat neuf p50 {num(c.bench.s1_cold_p50_ms, 1)} ms · p95 {num(c.bench.s1_cold_p95_ms ?? 0, 1)} ms{c.bench.s1_cold_prefill_tokens ? ` · ${num(c.bench.s1_cold_prefill_tokens)} tokens` : ""}
          </span>
        {/if}
        {#if c.bench.s1_warm_p50_ms != null}
          <span class="chip s1" title={c.bench.s1_series?.s1_warm ?? ""}>relu p50 {num(c.bench.s1_warm_p50_ms, 1)} ms · p95 {num(c.bench.s1_warm_p95_ms ?? 0, 1)} ms</span>
        {/if}
        {#if c.bench.demo}<span class="chip warn">faux serveurs (demo)</span>{/if}
      </div>
    {/if}
    {#if logs}
      <div class="logs">
        <div class="lh"><span class="mono">{logs.name}.log</span><button class="icon-btn" onclick={() => (logs = null)}><X size={13} /></button></div>
        <pre class="mono">{logs.lines.join("\n") || "(vide)"}</pre>
      </div>
    {/if}
  </section>

  {#each [["s2", "Cerveau · System Two", "Le modele qui raisonne, code et appelle les outils."], ["s1", "Classifieur · System One", "Le decideur type Jev : jugements probabilistes, sans generation ; calibrez-le pour des pourcentages honnetes."]] as [role, title, sub] (role)}
    <section class="card cat rise">
      <div class="ph"><div><div class="panel-title">{title}</div><p class="faint sub">{sub}</p></div></div>
      {#each c.catalog.models.filter((m) => m.role === role) as m (m.id)}
        {@const inst = c.installed.models[m.id]?.installed}
        {@const active = (role === "s2" ? (c.runtime.plan ?? plan).s2.model_id : (c.runtime.plan ?? plan).s1?.model_id) === m.id}
        <div class="model" class:active>
          <div class="mi">
            <div class="mt">
              <b>{m.label}</b>{#each m.tags as t}<span class="chip {t === 'recommande' ? 's1' : ''}">{t}</span>{/each}{#if active}<span class="chip ok">actif</span>{/if}
              {#if role === "s1" && inst}
                {@const cal = app.calibrations[m.id]}
                {#if cal && !cal.error}
                  <span class="chip ok" title="Temperature par primitive et seuils par question ({cal.source ?? '?'})">calibre{cal.n ? ` · ${cal.n} ex.` : ""}</span>
                {:else}
                  <span class="chip warn" title={cal?.error ?? "Pourcentages bruts : lancez Calibrer quand ce classifieur tourne, ou importez une calibration."}>non calibre</span>
                {/if}
              {/if}
            </div>
            <p class="faint">{m.note}{m.quality ? ` · ${m.quality}` : ""}{m.custom && m.path ? ` · ${m.path}` : ""}</p>
            {@render dlrow(m.id)}
          </div>
          <div class="ma">
            <span class="size tabnum">{gb(m.size_gb)}</span>
            {#if inst}
              <span class="chip ok"><Check size={12} /> installe</span>
              {#if role === "s1" && app.activeS1 === m.id}
                <button class="btn sm" onclick={() => app.calibrateS1()} disabled={!!app.calibrating}
                        title="Lit les graines etiquetees livrees avec ce classifieur et ajuste temperature + seuils (fichier propre a ce modele)">
                  {#if app.calibrating}<span class="spinner"></span> {app.calibrating.total ? `${app.calibrating.done}/${app.calibrating.total}` : "…"}{:else}<Target size={14} /> Calibrer{/if}
                </button>
              {/if}
              {#if !active && (role === "s2" ? c.settings.s2_model : c.settings.s1_model) === m.id}
                <span class="chip" title="Choisi dans les reglages : applique au prochain demarrage des modeles">choisi</span>
              {:else if !active}<button class="btn sm ghost" onclick={() => app.updateSettings(role === "s2" ? { s2_model: m.id } : { s1_model: m.id })}>Utiliser</button>{/if}
              <button class="icon-btn" title={m.custom ? "Retirer de la liste (le fichier GGUF est conserve)" : "Supprimer"} onclick={() => removeModel(m.id)}><Trash2 size={14} /></button>
            {:else if !jobs(m.id).length}
              <button class="btn sm" onclick={() => app.install([m.id])}><Download size={14} /> Installer</button>
            {/if}
          </div>
        </div>
      {/each}
    </section>
  {/each}

  <div class="grid">
    <section class="card cat rise">
      <div class="ph"><div><div class="panel-title">Voix</div><p class="faint sub">Sur CPU (sherpa-onnx) : la VRAM reste au modele.</p></div><Mic size={16} class="faint" /></div>
      {#each c.catalog.voice as v (v.id)}
        <div class="model">
          <div class="mi"><div class="mt"><b>{v.label}</b><span class="chip">{v.kind.toUpperCase()}</span></div>{#if v.note}<p class="faint">{v.note}</p>{/if}{@render dlrow(v.id)}</div>
          <div class="ma">
            <span class="size tabnum">{num(v.size_mb)} Mo</span>
            {#if c.installed.voice[v.id]}<span class="chip ok"><Check size={12} /> installe</span><button class="icon-btn" title="Retirer" onclick={() => app.removeInstalled(v.id)}><Trash2 size={14} /></button>
            {:else if !jobs(v.id).length}<button class="btn sm" onclick={() => app.install([v.id])}><Download size={14} /> Installer</button>{/if}
          </div>
        </div>
      {/each}
    </section>

    <section class="card cat rise">
      <div class="ph"><div><div class="panel-title">Runtime et modeles personnels</div><p class="faint sub">llama.cpp (fork PrismML), choisi selon votre GPU.</p></div><Server size={16} class="faint" /></div>
      <div class="model">
        <div class="mi">
          <div class="mt"><b>llama-server</b>{#if c.installed.runtime}<span class="chip">{c.installed.runtime.backend}{c.installed.runtime.cuda ? ` ${c.installed.runtime.cuda}` : ""}</span>{/if}</div>
          <p class="faint mono">{c.installed.runtime?.version || (c.installed.custom_server ? "binaire personnalise" : "non installe")}</p>
          {@render dlrow("runtime")}
        </div>
        <div class="ma">
          {#if c.installed.runtime || c.installed.custom_server}<span class="chip ok"><Check size={12} /> pret</span>{/if}
          {#if !jobs("runtime").length}<button class="btn sm" onclick={() => app.install(["runtime"])}><Download size={14} /> {c.installed.runtime ? "Reinstaller" : "Installer"}</button>{/if}
        </div>
      </div>
      <div class="import">
        <p class="faint">Importer un GGUF (par ex. votre clone entraine sur A100) :</p>
        <div class="row">
          <input class="input mono" placeholder="C:\chemin\vers\jev-clone-Q8_0.gguf" bind:value={importPath} />
          <select class="select" bind:value={importRole} style="width:150px"><option value="s1">Classifieur</option><option value="s2">Cerveau</option></select>
          <button class="btn" onclick={doImport} disabled={!importPath.trim()}><Upload size={14} /> Importer</button>
        </div>
        {#each missingCustom as [id, v] (id)}
          <div class="warnbox">
            <TriangleAlert size={14} />
            <span class="wtxt">GGUF importe introuvable ({id}) : <span class="mono">{v.path}</span></span>
            <button class="icon-btn" title="Retirer" onclick={() => removeModel(id)}><Trash2 size={13} /></button>
          </div>
        {/each}
        <p class="faint cal">Importer une calibration (calibration.json du notebook ou d'une autre machine) pour un classifieur :</p>
        <div class="row">
          <input class="input mono" placeholder="C:\chemin\vers\calibration.json" bind:value={calPath} />
          <select class="select" value={calTarget} onchange={(e) => (calModel = e.currentTarget.value)} style="width:150px" disabled={!s1Installed.length}>
            {#each s1Installed as m (m.id)}<option value={m.id}>{m.label}</option>{/each}
          </select>
          <button class="btn" onclick={doImportCalibration} disabled={!calPath.trim() || !calTarget}><Upload size={14} /> Importer</button>
        </div>
      </div>
    </section>
  </div>
</div>

<style>
  .page { flex: 1; overflow-y: auto; padding: 22px 26px 40px; display: flex; flex-direction: column; gap: 16px; max-width: 1240px; width: 100%; margin: 0 auto; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid.top { grid-template-columns: minmax(300px, 0.8fr) 1.2fr; }
  .card { padding: 18px 20px; }
  .ph { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
  .sub { margin: 4px 0 0; font-size: 12.5px; }
  .gpu { display: flex; gap: 14px; align-items: center; margin: 14px 0; }
  .gicon { width: 48px; height: 48px; border-radius: 14px; display: grid; place-items: center; background: linear-gradient(135deg, var(--s2-soft), var(--s1-soft)); color: var(--text); border: 1px solid var(--line-2); }
  h2 { margin: 0; font-size: 20px; letter-spacing: -0.02em; font-weight: 650; }
  h3 { margin: 0 0 14px; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .gpu p { margin: 3px 0 0; font-size: 12.5px; }
  .badges { display: flex; flex-wrap: wrap; gap: 6px; }
  .warnbox { display: flex; gap: 8px; align-items: flex-start; margin-top: 12px; padding: 10px 12px; border-radius: 10px; background: var(--warn-soft); color: var(--warn); font-size: 12.5px; line-height: 1.45; }
  .warnbox > :global(svg), .warnbox > button { flex: none; }
  .warnbox > :global(svg) { margin-top: 2px; }
  .warnbox > .btn { margin: -3px 0; }
  .wtxt { flex: 1; min-width: 0; overflow-wrap: anywhere; }
  .warnbox.ram { background: var(--err-soft); color: var(--err); }
  .kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0 10px; }
  .kpis > div { display: flex; flex-direction: column; gap: 3px; padding: 10px 12px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); }
  .kpis span { font-size: 11.5px; }
  .kpis b { font-size: 14px; font-weight: 620; display: inline-flex; align-items: center; gap: 6px; }
  .s1c { color: var(--s1); } .s2c { color: var(--s2); }
  .notes { margin: 6px 0 14px; padding-left: 18px; font-size: 12.5px; color: var(--text-2); line-height: 1.55; }
  .acts { display: flex; gap: 6px; }
  .msg { margin: -4px 0 10px; font-size: 12.5px; }
  .msg.err { color: var(--err); }
  .servers { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .srv { display: flex; align-items: center; gap: 12px; padding: 12px 14px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); }
  .sic { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; flex: none; }
  .sic.s2 { background: var(--s2-soft); color: var(--s2); } .sic.s1 { background: var(--s1-soft); color: var(--s1); }
  .st { flex: 1; display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .st b { font-size: 13px; font-weight: 600; }
  .st span { font-size: 12px; display: flex; align-items: center; gap: 7px; }
  .bench { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
  .logs { margin-top: 12px; border-radius: 12px; border: 1px solid var(--line); overflow: hidden; }
  .lh { display: flex; justify-content: space-between; align-items: center; padding: 4px 6px 4px 12px; background: var(--surface-2); font-size: 12px; }
  .logs pre { margin: 0; padding: 10px 12px; max-height: 260px; overflow: auto; font-size: 11.5px; background: var(--bg-2); color: var(--text-2); white-space: pre-wrap; }
  .model { display: flex; gap: 14px; align-items: center; padding: 12px 0; border-top: 1px solid var(--line); }
  .model.active .mt b { color: var(--text); }
  .mi { flex: 1; min-width: 0; }
  .mt { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .mt b { font-size: 13.5px; font-weight: 600; margin-right: 4px; }
  .mt .chip { height: 20px; font-size: 11px; }
  .mi p { margin: 4px 0 0; font-size: 12px; line-height: 1.45; }
  .ma { display: flex; align-items: center; gap: 8px; flex: none; }
  .size { font-size: 12px; color: var(--text-3); min-width: 56px; text-align: right; }
  .dl { margin-top: 10px; }
  .dlm { display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 11.5px; color: var(--text-3); margin-top: 4px; }
  .dlm .icon-btn { width: 22px; height: 22px; }
  .import { border-top: 1px solid var(--line); padding-top: 12px; }
  .import p { margin: 0 0 8px; font-size: 12.5px; }
  .import p.cal { margin-top: 14px; }
  .import .row { display: flex; gap: 8px; }
  @media (max-width: 1000px) { .grid, .grid.top, .servers { grid-template-columns: 1fr; } .kpis { grid-template-columns: repeat(2, 1fr); } }
</style>
