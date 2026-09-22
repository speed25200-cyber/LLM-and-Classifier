<script lang="ts">
  // Deux arcs : System Two (exterieur, la pensee) et System One (interieur, l'eclair), un seul coeur.
  let { size = 22, animated = false, thinking = false }: { size?: number; animated?: boolean; thinking?: boolean } = $props();
  const id = `g${Math.random().toString(36).slice(2, 8)}`;
</script>

<svg class="logo" class:animated class:thinking width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
  <defs>
    <linearGradient {id} x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#8b7cff" />
      <stop offset="0.6" stop-color="#5fb4ff" />
      <stop offset="1" stop-color="#3ee6d0" />
    </linearGradient>
  </defs>
  <g class="outer"><path d="M12 42a20 20 0 0 1 40 0" fill="none" stroke="url(#{id})" stroke-width="5.5" stroke-linecap="round" /></g>
  <g class="inner"><path d="M21.5 42a10.5 10.5 0 0 1 21 0" fill="none" stroke="#3ee6d0" stroke-width="5.5" stroke-linecap="round" /></g>
  <circle cx="32" cy="42" r="3.6" fill="currentColor" />
</svg>

<style>
  .logo { color: var(--text); flex: none; overflow: visible; }
  .outer, .inner { transform-origin: 32px 42px; }
  .animated .outer { animation: sway 6s var(--ease) infinite; }
  .animated .inner { animation: sway 6s var(--ease) infinite reverse; }
  .thinking .outer { animation: think 1.6s var(--ease) infinite; }
  .thinking .inner { animation: think 1.6s var(--ease) 0.2s infinite; }
  @keyframes sway { 0%, 100% { transform: rotate(-4deg); } 50% { transform: rotate(4deg); } }
  @keyframes think { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
</style>
