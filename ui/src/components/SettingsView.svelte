<script lang="ts">
  import { app } from "../lib/store.svelte";
  import { toggleHandsfree } from "../lib/voice";
  import type { Settings } from "../lib/types";

  let { onTheme, theme }: { onTheme: () => void; theme: string } = $props();
  const s = $derived(app.core!.settings);
  const voices = $derived(app.core!.catalog.voice);

  function set(patch: Partial<Settings> | Record<string, unknown>) {
    app.updateSettings(patch);
  }
  function setVoice(patch: Record<string, unknown>) {
    app.updateSettings({ voice: patch });
  }
  function num(e: Event) {
    return Number((e.target as HTMLInputElement).value);
  }
  function val(e: Event) {
    return (e.target as HTMLInputElement).value;
  }
</script>

{#snippet toggle(on: boolean, onchange: () => void, label: string, desc: string)}
  <button class="row tog" onclick={onchange} role="switch" aria-checked={on}>
    <div class="lab"><b>{label}</b><span>{desc}</span></div>
    <span class="switch" class:on></span>
  </button>
{/snippet}

<div class="page">
  <section class="card rise">
    <div class="panel-title">General</div>
    <div class="row">
      <div class="lab"><b>Langue de l'agent et de la voix</b><span>Reconnaissance vocale, voix de synthese, reponses.</span></div>
      <div class="seg"><button class:on={s.language === "fr"} onclick={() => set({ language: "fr", voice: { tts_voice: "tts-fr-siwis" } })}>Francais</button><button class:on={s.language === "en"} onclick={() => set({ language: "en", voice: { tts_voice: "tts-en-lessac" } })}>English</button></div>
    </div>
    <div class="row">
      <div class="lab"><b>Theme</b><span>Sombre (obsidienne) ou clair.</span></div>
      <div class="seg"><button class:on={theme === "dark"} onclick={() => theme !== "dark" && onTheme()}>Sombre</button><button class:on={theme === "light"} onclick={() => theme !== "light" && onTheme()}>Clair</button></div>
    </div>
    <div class="row col">
      <div class="lab"><b>Espace de travail par defaut</b><span>Dossier ou Prophet cree et modifie des fichiers (une session peut en avoir un autre).</span></div>
      <input class="input mono" value={s.workspace} onchange={(e) => set({ workspace: val(e) })} />
    </div>
  </section>

  <section class="card rise">
    <div class="panel-title">Agent</div>
    <div class="row">
      <div class="lab"><b>Autorisations</b><span>Smart : le classifieur juge chaque action (~100 ms) et ne demande que si elle est risquee.</span></div>
      <div class="seg">
        {#each [["smart", "Smart"], ["ask", "Toujours demander"], ["auto", "Jamais"]] as [k, l] (k)}<button class:on={s.permission_mode === k} onclick={() => set({ permission_mode: k })}>{l}</button>{/each}
      </div>
    </div>
    <div class="row">
      <div class="lab"><b>Effort de reflexion</b><span>Auto : budget choisi par le classifieur selon le risque et la difficulte.</span></div>
      <div class="seg">
        {#each [["auto", "Auto"], ["fast", "Rapide"], ["deep", "Profond"]] as [k, l] (k)}<button class:on={s.effort === k} onclick={() => set({ effort: k })}>{l}</button>{/each}
      </div>
    </div>
    {@render toggle(s.browser_tool, () => set({ browser_tool: !s.browser_tool }), "Outil navigateur (computer use)", "Bonsai pilote un navigateur, le classifieur decide chaque pas (Playwright requis).")}
  </section>

  <section class="card rise">
    <div class="panel-title">Voix</div>
    {@render toggle(s.voice.enabled, () => setVoice({ enabled: !s.voice.enabled }), "Commandes vocales", "Push-to-talk : maintenir le micro ou Ctrl+Maj+Espace.")}
    {@render toggle(app.voice.handsfree, () => toggleHandsfree(), "Mains libres", `Ecoute continue ; dites « ${s.voice.wake_word}, … » pour parler a l'agent.`)}
    {@render toggle(s.voice.speak_responses, () => setVoice({ speak_responses: !s.voice.speak_responses }), "Lire les reponses a voix haute", "Synthese locale, phrase par phrase.")}
    <div class="row">
      <div class="lab"><b>Reconnaissance</b><span>Parakeet : 25 langues, tres rapide sur CPU.</span></div>
      <select class="select" style="width:280px" value={s.voice.stt_model} onchange={(e) => setVoice({ stt_model: val(e) })}>
        {#each voices.filter((v) => v.kind === "stt") as v (v.id)}<option value={v.id}>{v.label}</option>{/each}
      </select>
    </div>
    <div class="row">
      <div class="lab"><b>Voix de synthese</b><span>Sans modele installe, la voix du systeme prend le relais.</span></div>
      <select class="select" style="width:280px" value={s.voice.tts_voice} onchange={(e) => setVoice({ tts_voice: val(e) })}>
        {#each voices.filter((v) => v.kind === "tts") as v (v.id)}<option value={v.id}>{v.label}</option>{/each}
      </select>
    </div>
    <div class="row">
      <div class="lab"><b>Mot d'eveil</b><span>Prononce en debut de phrase en mode mains libres.</span></div>
      <input class="input" style="width:200px" value={s.voice.wake_word} onchange={(e) => setVoice({ wake_word: val(e) })} />
    </div>
    <div class="row">
      <div class="lab"><b>Seuil des commandes vocales · {Math.round(s.voice.command_threshold * 100)} %</b><span>Probabilite calibree minimale pour executer une commande ; en dessous, la phrase part a l'agent.</span></div>
      <input type="range" min="0.5" max="0.98" step="0.01" value={s.voice.command_threshold} onchange={(e) => setVoice({ command_threshold: num(e) })} />
    </div>
    <div class="row">
      <div class="lab"><b>Debit de la voix · {s.voice.speed.toFixed(2)}x</b><span></span></div>
      <input type="range" min="0.7" max="1.6" step="0.05" value={s.voice.speed} onchange={(e) => setVoice({ speed: num(e) })} />
    </div>
  </section>

  <section class="card rise">
    <div class="panel-title">Performance</div>
    {@render toggle(s.vram_saver, () => set({ vram_saver: !s.vram_saver }), "Economie de VRAM (application de bureau)", "L'interface est rendue par le CPU : chaque Mio de la carte reste au modele. Prend effet au prochain lancement.")}
    {@render toggle(s.reduce_motion, () => set({ reduce_motion: !s.reduce_motion }), "Reduire les animations", "Interface encore plus sobre en ressources.")}
    {@render toggle(s.autostart_models, () => set({ autostart_models: !s.autostart_models }), "Demarrer les modeles au lancement", "Bonsai se charge en ~10-30 s selon le disque.")}
    <div class="row">
      <div class="lab"><b>Contexte force</b><span>0 = automatique (planificateur VRAM). En tokens.</span></div>
      <input class="input" type="number" min="0" step="1024" style="width:140px" value={s.ctx_override} onchange={(e) => set({ ctx_override: num(e) })} />
    </div>
  </section>

  <section class="card rise">
    <div class="panel-title">Avance</div>
    <div class="row col"><div class="lab"><b>Miroir Hugging Face</b><span>Pour les reseaux ou huggingface.co est lent ou bloque.</span></div><input class="input mono" value={s.hf_endpoint} onchange={(e) => set({ hf_endpoint: val(e) })} /></div>
    <div class="row col"><div class="lab"><b>llama-server personnalise</b><span>Chemin d'un binaire compile vous-meme (ex. build sm_120). Vide = runtime installe.</span></div><input class="input mono" value={s.llama_server_path} onchange={(e) => set({ llama_server_path: val(e) })} /></div>
    <div class="row"><div class="lab"><b>Release du runtime</b><span>Tag du fork PrismML llama.cpp.</span></div><input class="input mono" style="width:260px" value={s.runtime_tag} onchange={(e) => set({ runtime_tag: val(e) })} /></div>
    <div class="row"><div class="lab"><b>Ports</b><span>Bonsai / classifieur (127.0.0.1 uniquement).</span></div>
      <div class="ports"><input class="input" type="number" value={s.s2_port} onchange={(e) => set({ s2_port: num(e) })} /><input class="input" type="number" value={s.s1_port} onchange={(e) => set({ s1_port: num(e) })} /></div>
    </div>
    <div class="row"><div class="lab"><b>Donnees</b><span class="mono">{app.core!.data_dir}</span></div><span class="faint">v{app.core!.version}</span></div>
    <div class="row"><div class="lab"><b>API locale compatible</b><span class="mono">/v1/chat/completions · /v1/systemone (jeton requis)</span></div></div>
  </section>
</div>

<style>
  .page { flex: 1; overflow-y: auto; padding: 22px 26px 40px; display: flex; flex-direction: column; gap: 16px; max-width: 860px; width: 100%; margin: 0 auto; }
  .card { padding: 16px 20px 6px; }
  .row { width: 100%; display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 12px 0; border-top: 1px solid var(--line); text-align: left; }
  .card > .panel-title + .row { border-top: 0; }
  .row.col { flex-direction: column; align-items: stretch; gap: 8px; }
  .lab { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .lab b { font-size: 13.5px; font-weight: 580; }
  .lab span { font-size: 12.5px; color: var(--text-3); line-height: 1.45; overflow-wrap: anywhere; }
  .tog:hover .switch:not(.on) { background: var(--line-3); }
  .ports { display: flex; gap: 8px; width: 220px; }
  input[type="range"] { width: 220px; accent-color: var(--s2); }
  .panel-title { margin-bottom: 4px; }
</style>
