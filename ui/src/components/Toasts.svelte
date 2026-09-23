<script lang="ts">
  import { CircleAlert, CircleCheck, Info, Mic, TriangleAlert } from "@lucide/svelte";
  import { app } from "../lib/store.svelte";
  const ICON = { info: Info, ok: CircleCheck, warn: TriangleAlert, error: CircleAlert, voice: Mic };
</script>

<div class="toasts" aria-live="polite">
  {#each app.toasts as t (t.id)}
    {@const I = ICON[t.kind]}
    <div class="toast {t.kind}" style="--ttl:{t.ttl}ms">
      <span class="ic"><I size={15} /></span>
      <div class="tx"><b>{t.title}</b>{#if t.body}<span title={t.body}>{t.body}</span>{/if}</div>
      <i class="life"></i>
    </div>
  {/each}
</div>

<style>
  .toasts { position: fixed; right: 16px; bottom: calc(var(--statusbar-h) + 14px); z-index: 60; display: flex; flex-direction: column; gap: 8px; width: min(380px, calc(100vw - 32px)); pointer-events: none; }
  .toast { position: relative; overflow: hidden; display: flex; gap: 11px; padding: 11px 14px; border-radius: 14px; background: var(--surface-2); border: 1px solid var(--line-2); box-shadow: var(--shadow-3); animation: pop 300ms var(--ease-spring) both; pointer-events: auto; }
  @keyframes pop { from { opacity: 0; transform: translateY(10px) scale(0.97); } }
  .ic { flex: none; margin-top: 1px; color: var(--text-2); }
  .ok .ic { color: var(--ok); } .warn .ic { color: var(--warn); } .error .ic { color: var(--err); } .voice .ic { color: var(--s1); }
  .tx { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
  .tx b { font-size: 13px; font-weight: 600; }
  /* messages du moteur (cause d'un arret, mode mono) : 4 lignes au plus, le texte entier en infobulle */
  .tx span { font-size: 12px; color: var(--text-2); overflow-wrap: anywhere; display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 4; line-clamp: 4; overflow: hidden; }
  .life { position: absolute; left: 0; bottom: 0; height: 2px; width: 100%; background: currentColor; opacity: 0.25; transform-origin: left; animation: life var(--ttl) linear both; }
  .ok .life { color: var(--ok); } .warn .life { color: var(--warn); } .error .life { color: var(--err); } .voice .life { color: var(--s1); }
  @keyframes life { to { transform: scaleX(0); } }
</style>
