<script lang="ts">
  import type { Plan } from "../lib/types";
  import { mibToGib } from "../lib/format";

  let { plan, height = 14, legend = true }: { plan: Plan; height?: number; legend?: boolean } = $props();
  const SEG = [
    { k: "other", label: "Bureau et autres applis", c: "var(--text-4)" },
    { k: "s2_weights", label: "Bonsai · poids", c: "var(--s2)" },
    { k: "s2_runtime", label: "Bonsai · calcul", c: "color-mix(in srgb, var(--s2) 60%, #5fb4ff)" },
    { k: "s2_kv", label: "Cache KV (contexte)", c: "#5fb4ff" },
    { k: "mmproj", label: "Vision", c: "#ff8fa3" },
    { k: "s1_weights", label: "Classifieur", c: "var(--s1)" },
    { k: "s1_runtime", label: "Classifieur · calcul", c: "color-mix(in srgb, var(--s1) 70%, #fff)" },
    { k: "s1_kv", label: "Classifieur · KV", c: "color-mix(in srgb, var(--s1) 45%, #5fb4ff)" },
    { k: "reserve", label: "Marge de securite", c: "repeating-linear-gradient(135deg, var(--surface-4) 0 4px, transparent 4px 8px)" },
  ];
  const cpu = $derived(plan.backend === "cpu");
  const total = $derived(cpu ? plan.budget.ram_total_mib : plan.budget.total);
  const segs = $derived(
    cpu
      ? [{ k: "ram", label: "Modeles en RAM", c: "var(--s2)", v: plan.budget.ram_needed_mib }]
      : SEG.map((s) => ({ ...s, v: plan.budget[s.k] ?? 0 })).filter((s) => s.v > 1),
  );
  const used = $derived(segs.reduce((a, s) => a + s.v, 0));
</script>

<div class="vram">
  <div class="track" style="height:{height}px">
    {#each segs as s (s.k)}
      <i style="flex-grow:{s.v}; background:{s.c}" title="{s.label} · {mibToGib(s.v)}"></i>
    {/each}
    <i class="free" style="flex-grow:{Math.max(0, total - used)}" title="Libre · {mibToGib(Math.max(0, total - used))}"></i>
  </div>
  {#if legend}
    <div class="legend">
      {#each segs as s (s.k)}
        <span><b style="background:{s.c}"></b>{s.label} <em class="tabnum">{mibToGib(s.v)}</em></span>
      {/each}
      <span><b class="freeb"></b>Libre <em class="tabnum">{mibToGib(Math.max(0, total - used))}</em></span>
    </div>
  {/if}
</div>

<style>
  .track { display: flex; gap: 2px; border-radius: 8px; overflow: hidden; background: var(--surface-3); }
  .track i { min-width: 2px; transition: flex-grow 500ms var(--ease); }
  .track i:first-child { border-radius: 8px 0 0 8px; }
  .free { background: transparent; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 10px; font-size: 11.5px; color: var(--text-2); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend b { width: 9px; height: 9px; border-radius: 3px; display: inline-block; }
  .legend em { font-style: normal; color: var(--text-3); }
  .freeb { border: 1px solid var(--line-3); }
</style>
