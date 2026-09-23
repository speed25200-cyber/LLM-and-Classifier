<script lang="ts">
  import { ArrowLeft, ArrowRight, Brain, Check, Cpu, Download, HardDrive, Rocket, TriangleAlert, Zap, Mic } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  import { bytes, eta, gb, mibToGib, num } from "../lib/format";
  import type { Priority } from "../lib/types";
  import Logo from "./Logo.svelte";
  import VramBar from "./VramBar.svelte";

  const c = $derived(app.core!);
  const g = $derived(c.hardware.primary_gpu);
  const plan = $derived(c.plan);
  let step = $state(0);
  let started = $state(false);
  let chosen = $state<Record<string, boolean>>({});

  $effect(() => {
    for (const r of c.recommended) if (chosen[r.id] === undefined) chosen[r.id] = true;
  });

  const checks = $derived.by(() => {
    const out: { ok: boolean | null; label: string; detail: string }[] = [];
    if (g && g.vendor === "nvidia") {
      out.push({ ok: true, label: "GPU NVIDIA", detail: g.name.replace("NVIDIA GeForce ", "") });
      out.push({ ok: g.vram_total_mib >= 7500, label: "VRAM", detail: mibToGib(g.vram_total_mib) + (g.vram_total_mib >= 7500 ? " : Bonsai 2 27B tient entierement" : " : profil reduit") });
      const cuda = parseFloat(g.cuda_version || "0");
      out.push({ ok: !g.is_blackwell || cuda >= 12.8, label: "Pilote", detail: `${g.driver} · CUDA ${g.cuda_version}${g.is_blackwell && cuda < 12.8 ? " : pilote >= 570 requis pour Blackwell" : ""}` });
    } else {
      out.push({ ok: null, label: "GPU", detail: "aucun GPU NVIDIA : mode CPU (plus lent)" });
    }
    out.push({ ok: c.hardware.ram_total_gib >= 15, label: "Memoire", detail: `${num(c.hardware.ram_total_gib)} Gio de RAM` });
    const need = c.recommended.reduce((a, r) => a + r.size_gb, 0);
    out.push({ ok: c.installed.free_disk_gb > need + 2, label: "Disque", detail: `${num(c.installed.free_disk_gb, 1)} Go libres, ${num(need, 1)} Go a telecharger` });
    return out;
  });

  const selected = $derived(c.recommended.filter((r) => chosen[r.id]));
  const total = $derived(selected.reduce((a, r) => a + r.size_gb, 0));
  const jobs = $derived(Object.values(app.downloads).filter((j) => selected.some((r) => r.id === j.group)));
  const dlDone = $derived(jobs.reduce((a, j) => a + j.done_bytes, 0));
  const dlSize = $derived(jobs.reduce((a, j) => a + (j.size || 0), 0));
  const speed = $derived(jobs.reduce((a, j) => a + (j.status === "running" ? j.speed_bps : 0), 0));
  const allDone = $derived(c.recommended.filter((r) => r.role !== "voice").length === 0);
  const ready = $derived(app.ready);

  async function install() {
    started = true;
    await app.install(selected.map((r) => r.id));
  }
  async function finish() {
    await app.updateSettings({ onboarding_done: true });
    app.wizardOpen = false;
  }
  const STEPS = ["Bienvenue", "Machine", "Configuration", "Installation", "Pret"];
  const PRIO: [Priority, string][] = [["equilibre", "Equilibre"], ["contexte", "Contexte"], ["vitesse", "Vitesse"]];
</script>

<div class="wiz fade-in">
  <div class="panel rise">
    <aside class="steps">
      <div class="brand"><Logo size={22} /> Prophet Studio</div>
      {#each STEPS as s, i (s)}
        <div class="st" class:on={i === step} class:done={i < step}><span class="n">{#if i < step}<Check size={12} />{:else}{i + 1}{/if}</span>{s}</div>
      {/each}
      <button class="skip" onclick={finish}>Passer l'assistant</button>
    </aside>

    <div class="main">
      {#if step === 0}
        <div class="hero">
          <Logo size={72} animated />
          <h1>Deux cerveaux. <span class="grad-text">Une seule machine.</span></h1>
          <p class="muted">Prophet Studio fusionne un <b class="s1c">classifieur type Jev</b> qui decide en quelques dizaines de millisecondes et <b class="s2c">Bonsai 2 27B</b>, un modele de 27 milliards de parametres qui tient dans 6 Go. Code, applications, commandes, web, voix : tout tourne en local.</p>
          <div class="duo">
            <div><Zap size={18} class="s1c" /><b>System One</b><span>Decisions calibrees en une passe : quel outil, quel risque, combien reflechir, commande vocale ou demande.</span></div>
            <div><Brain size={18} class="s2c" /><b>System Two</b><span>Bonsai 2 27B (98 % de Qwen3.8-27B) : raisonnement, code, outils, vision.</span></div>
          </div>
        </div>
      {:else if step === 1}
        <h2>Votre machine</h2>
        <p class="muted">Detection automatique ; le planificateur en deduit la meilleure configuration.</p>
        <div class="checks">
          {#each checks as ck (ck.label)}
            <div class="ck"><span class="ci {ck.ok === true ? 'ok' : ck.ok === false ? 'bad' : ''}">{#if ck.ok === true}<Check size={14} />{:else if ck.ok === false}<TriangleAlert size={14} />{:else}<Cpu size={14} />{/if}</span><b>{ck.label}</b><span>{ck.detail}</span></div>
          {/each}
        </div>
        {#each c.hardware.warnings as w}<p class="warn"><TriangleAlert size={14} /> {w}</p>{/each}
      {:else if step === 2}
        <h2>Configuration optimale</h2>
        <div class="seg">{#each PRIO as [k, l] (k)}<button class:on={c.settings.priority === k} onclick={() => app.updateSettings({ priority: k })}>{l}</button>{/each}</div>
        <h3>{plan.title}</h3>
        <VramBar {plan} />
        <div class="kp">
          <span class="chip s2"><Brain size={12} /> {plan.expected.s2.tok_s ? `${plan.expected.s2.tok_s[0]}-${plan.expected.s2.tok_s[1]} tok/s estimes` : "vitesse a mesurer"}</span>
          <span class="chip s1"><Zap size={12} /> decisions {plan.expected.s1_ms[0]}-{plan.expected.s1_ms[1]} ms</span>
          <span class="chip">contexte {Math.round(plan.s2.ctx / 1024)} k</span>
        </div>
        <ul class="notes">{#each plan.notes as n}<li>{n}</li>{/each}</ul>
      {:else if step === 3}
        <h2>Installation</h2>
        {#if c.recommended.length === 0}
          <p class="muted"><Check size={14} /> Tout est deja installe.</p>
        {:else}
          <p class="muted">Telechargements reprenables et verifies (SHA-256). Vous pouvez continuer pendant qu'ils avancent.</p>
          <div class="items">
            {#each c.recommended as r (r.id)}
              {@const js = Object.values(app.downloads).filter((j) => j.group === r.id)}
              {@const p = js.reduce((a, j) => a + j.done_bytes, 0) / Math.max(1, js.reduce((a, j) => a + (j.size || 0), 0))}
              <label class="item">
                <input type="checkbox" bind:checked={chosen[r.id]} disabled={started} />
                <span class="ri">{#if r.role === "voice"}<Mic size={14} />{:else if r.id === "runtime"}<HardDrive size={14} />{:else if r.role === "s1"}<Zap size={14} />{:else}<Brain size={14} />{/if}</span>
                <span class="rl">{r.label}</span>
                {#if js.length}<span class="mini"><i style="transform: scaleX({js.every((j) => j.status === 'done') ? 1 : p})"></i></span>{/if}
                <span class="rs tabnum">{gb(r.size_gb)}</span>
              </label>
            {/each}
          </div>
          {#if started}
            <div class="global">
              <div class="bar"><i style="transform: scaleX({dlSize ? dlDone / dlSize : 0})"></i></div>
              <span class="faint tabnum">{bytes(dlDone)} / {bytes(dlSize)} {speed ? `· ${bytes(speed)}/s · ${eta((dlSize - dlDone) / speed)}` : ""}</span>
            </div>
          {:else}
            <button class="btn accent lg" onclick={install} disabled={!selected.length}><Download size={16} /> Tout installer · {num(total, 1)} Go</button>
          {/if}
        {/if}
      {:else}
        <div class="hero">
          <div class="rk"><Rocket size={30} /></div>
          <h2>{ready ? "Tout est pret." : allDone ? "Demarrage des modeles…" : "Presque pret"}</h2>
          <p class="muted">
            {#if ready}Bonsai et le classifieur tournent. Essayez « Cree une app minuteur » ou maintenez le micro pour parler.
            {:else if allDone}Chargement de Bonsai en memoire video (10 a 30 s).
            {:else}Les telechargements continuent en arriere-plan ; les modeles demarreront a la fin.{/if}
          </p>
          {#if !ready && allDone && c.runtime.state !== "starting"}<button class="btn" onclick={() => app.startRuntime()}>Demarrer les modeles</button>{/if}
          {#if c.runtime.state === "starting"}<div class="bar indeterminate" style="width:260px"><i></i></div>{/if}
        </div>
      {/if}

      <div class="nav">
        {#if step > 0}<button class="btn ghost" onclick={() => step--}><ArrowLeft size={15} /> Retour</button>{:else}<span></span>{/if}
        {#if step < 4}
          <button class="btn primary" onclick={() => (step === 3 && !started && c.recommended.length ? install().then(() => step++) : step++)}>
            {step === 3 && !started && c.recommended.length ? "Installer et continuer" : "Continuer"} <ArrowRight size={15} />
          </button>
        {:else}
          <button class="btn accent" onclick={finish}>Commencer <ArrowRight size={15} /></button>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  /* sous la barre de titre : fenetre deplacable, reduisible et fermable pendant l'accueil */
  .wiz { position: fixed; inset: var(--titlebar-h) 0 0 0; z-index: 70; background: rgba(4, 5, 8, 0.72); display: grid; place-items: center; padding: 24px; }
  .panel { width: min(980px, 100%); height: min(640px, 100%); display: grid; grid-template-columns: 230px 1fr; border-radius: 24px; background: var(--surface); box-shadow: var(--shadow-3); overflow: hidden; }
  .steps { background: var(--bg-2); border-right: 1px solid var(--line); padding: 22px 16px; display: flex; flex-direction: column; gap: 4px; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 650; font-size: 14px; margin: 0 6px 22px; }
  .st { display: flex; align-items: center; gap: 10px; padding: 9px 10px; border-radius: 10px; font-size: 13px; color: var(--text-3); }
  .st.on { background: var(--surface-2); color: var(--text); }
  .st.done { color: var(--text-2); }
  .n { width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center; font-size: 11px; font-weight: 650; background: var(--surface-3); }
  .st.on .n { background: var(--grad); color: #0b0c10; }
  .st.done .n { background: var(--ok-soft); color: var(--ok); }
  .skip { margin-top: auto; font-size: 12px; color: var(--text-3); padding: 8px 10px; text-align: left; border-radius: 8px; }
  .skip:hover { color: var(--text-2); background: var(--surface-2); }
  .main { padding: 34px 40px 24px; display: flex; flex-direction: column; overflow-y: auto; min-height: 0; }
  .hero { display: flex; flex-direction: column; align-items: center; text-align: center; gap: 14px; margin: auto 0; }
  h1 { margin: 8px 0 0; font-size: 34px; letter-spacing: -0.035em; font-weight: 700; line-height: 1.15; }
  h2 { margin: 0 0 6px; font-size: 24px; letter-spacing: -0.025em; font-weight: 660; }
  h3 { margin: 18px 0 12px; font-size: 15px; font-weight: 600; }
  .hero p { max-width: 600px; margin: 0; font-size: 14.5px; line-height: 1.6; }
  .s1c { color: var(--s1); } .s2c { color: var(--s2); }
  :global(.wiz .s1c) { color: var(--s1); } :global(.wiz .s2c) { color: var(--s2); }
  .duo { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; width: 100%; max-width: 640px; }
  .duo > div { display: flex; flex-direction: column; gap: 6px; text-align: left; padding: 16px; border-radius: 16px; background: var(--surface-2); border: 1px solid var(--line); }
  .duo b { font-size: 14px; }
  .duo span { font-size: 12.5px; color: var(--text-3); line-height: 1.5; }
  .checks { display: flex; flex-direction: column; gap: 8px; margin-top: 18px; }
  .ck { display: grid; grid-template-columns: 28px 110px 1fr; align-items: center; gap: 10px; padding: 12px 14px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); font-size: 13px; }
  .ck span:last-child { color: var(--text-2); }
  .ci { width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; background: var(--surface-3); color: var(--text-2); }
  .ci.ok { background: var(--ok-soft); color: var(--ok); }
  .ci.bad { background: var(--warn-soft); color: var(--warn); }
  .warn { display: flex; gap: 8px; color: var(--warn); font-size: 12.5px; background: var(--warn-soft); padding: 10px 12px; border-radius: 10px; margin: 12px 0 0; }
  .kp { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }
  .notes { margin: 12px 0 0; padding-left: 18px; font-size: 12.5px; color: var(--text-2); line-height: 1.6; }
  .items { display: flex; flex-direction: column; gap: 6px; margin: 16px 0; }
  .item { display: flex; align-items: center; gap: 12px; padding: 11px 14px; border-radius: 12px; background: var(--surface-2); border: 1px solid var(--line); font-size: 13px; cursor: pointer; }
  .item input { accent-color: var(--s2); width: 15px; height: 15px; }
  .ri { color: var(--text-3); display: grid; }
  .rl { flex: 1; }
  .rs { color: var(--text-3); font-size: 12px; min-width: 60px; text-align: right; }
  .mini { position: relative; width: 90px; height: 5px; border-radius: 5px; background: var(--surface-4); overflow: hidden; }
  .mini i { position: absolute; inset: 0; background: var(--grad); transform-origin: left; transition: transform 400ms var(--ease); }
  .global { display: flex; flex-direction: column; gap: 8px; }
  .rk { width: 72px; height: 72px; border-radius: 22px; display: grid; place-items: center; background: var(--grad); color: #0b0c10; }
  .nav { margin-top: auto; padding-top: 22px; display: flex; justify-content: space-between; }
  @media (max-width: 760px) { .panel { grid-template-columns: 1fr; } .steps { display: none; } .duo { grid-template-columns: 1fr; } }
</style>
