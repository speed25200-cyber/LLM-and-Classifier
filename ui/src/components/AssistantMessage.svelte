<script lang="ts">
  import { BadgeCheck, Copy, ListChecks, TriangleAlert, Volume2, Zap } from "@lucide/svelte";
  import type { AssistantItem } from "../lib/types";
  import { responseAlreadyShown } from "../lib/transcript";
  import { ms, num, pct } from "../lib/format";
  import { speaker } from "../lib/voice";
  import Logo from "./Logo.svelte";
  import Markdown from "./Markdown.svelte";
  import S1Strip from "./S1Strip.svelte";
  import ThinkingBlock from "./ThinkingBlock.svelte";
  import ToolCard from "./ToolCard.svelte";
  import PermissionCard from "./PermissionCard.svelte";

  let { item }: { item: AssistantItem } = $props();
  const live = $derived(item.status === "running");
  const visible = $derived(item.blocks.filter((b) => !(b.type === "tool" && b.name === "done")));
  const showResponse = $derived(!!item.response && item.status !== "running" && !responseAlreadyShown(item));
  const lastIdx = $derived(visible.length - 1);
  const final = $derived(item.response || item.blocks.filter((b) => b.type === "text").map((b) => (b as { text: string }).text).join("\n"));
  const nTools = $derived(item.blocks.filter((b) => b.type === "tool" && b.name !== "done").length);
  let copied = $state(false);

  function copy() {
    navigator.clipboard?.writeText(final);
    copied = true;
    setTimeout(() => (copied = false), 1400);
  }
</script>

<article class="msg" class:live>
  <div class="gutter"><Logo size={20} thinking={live} /></div>
  <div class="content">
    <div class="who">
      <b>Prophet</b>
      {#if item.plan_mode}<span class="chip s2"><ListChecks size={11} /> mode plan</span>{/if}
    </div>

    {#if item.s1}
      <S1Strip s1={item.s1} />
    {:else if live}
      <div class="analysing"><Zap size={13} /> <span class="shimmer">Le classifieur analyse la demande…</span></div>
    {/if}

    {#each visible as b, i (b.type === "tool" || b.type === "permission" ? b.id : `${b.type}-${i}`)}
      {#if b.type === "thinking"}
        <ThinkingBlock text={b.text} live={live && i === lastIdx && b.ms == null} duration={b.ms} />
      {:else if b.type === "text"}
        {#if b.text.trim()}<div class="text"><Markdown text={b.text} streaming={live && i === lastIdx} /></div>{/if}
      {:else if b.type === "tool"}
        <ToolCard {b} />
      {:else if b.type === "permission"}
        <PermissionCard {b} />
      {/if}
    {/each}

    {#if live && item.pending_tool}
      <div class="pending"><span class="spinner"></span> <span class="shimmer">Prepare {item.pending_tool}…</span></div>
    {/if}

    {#if showResponse}
      <div class="final rise"><Markdown text={item.response ?? ""} /></div>
    {/if}

    {#if item.status === "error"}
      <div class="error"><TriangleAlert size={14} /> {item.error}</div>
    {/if}

    {#if item.status === "done"}
      <footer class="meta tabnum">
        {#if item.verification != null}
          <span class="chip {item.verification >= 0.7 ? 'ok' : item.verification >= 0.4 ? '' : 'warn'}" title="Verification finale par le classifieur">
            <BadgeCheck size={12} /> verifie {pct(item.verification)}
          </span>
        {/if}
        {#if item.stopped_by === "cancelled"}<span class="chip warn">interrompu</span>{/if}
        <span class="faint">{ms(item.latency_ms)}</span>
        {#if item.stats?.tok_s}<span class="faint">· {num(item.stats.tok_s, 1)} tok/s</span>{/if}
        {#if item.stats?.tokens}<span class="faint">· {num(item.stats.tokens)} tokens</span>{/if}
        {#if nTools}<span class="faint">· {nTools} outil{nTools > 1 ? "s" : ""}</span>{/if}
        <span class="acts">
          <button class="icon-btn" title="Copier la reponse" onclick={copy}>{#if copied}<BadgeCheck size={14} />{:else}<Copy size={14} />{/if}</button>
          <button class="icon-btn" title="Lire a voix haute" onclick={() => speaker.speak(final)}><Volume2 size={14} /></button>
        </span>
      </footer>
    {/if}
  </div>
</article>

<style>
  .msg { display: grid; grid-template-columns: 28px 1fr; gap: 14px; padding: 6px 0 22px; content-visibility: auto; contain-intrinsic-size: auto 240px; }
  .gutter { padding-top: 2px; }
  .content { min-width: 0; }
  .who { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; font-size: 13px; }
  .who b { font-weight: 650; }
  .who .chip { height: 20px; font-size: 11px; }
  .analysing, .pending { display: flex; align-items: center; gap: 8px; font-size: 12.5px; color: var(--s1); margin: 4px 0 10px; }
  .pending { color: var(--s2); }
  .pending .spinner { width: 12px; height: 12px; }
  .text { margin: 4px 0 10px; }
  .final { margin-top: 12px; padding: 14px 18px; border-radius: 14px; background: linear-gradient(180deg, color-mix(in srgb, var(--s2) 5%, var(--surface)), var(--surface)); border: 1px solid var(--line); }
  .error { display: flex; gap: 8px; align-items: center; color: var(--err); background: var(--err-soft); padding: 10px 12px; border-radius: 10px; font-size: 13px; margin-top: 8px; }
  .meta { display: flex; align-items: center; gap: 8px; margin-top: 12px; font-size: 12px; flex-wrap: wrap; }
  .meta .chip { height: 22px; font-size: 11.5px; }
  .acts { display: flex; gap: 2px; margin-left: auto; opacity: 0; transition: opacity var(--t-fast); }
  .msg:hover .acts { opacity: 1; }
  .acts .icon-btn { width: 28px; height: 28px; }
</style>
