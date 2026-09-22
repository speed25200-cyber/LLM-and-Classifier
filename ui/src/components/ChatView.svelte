<script lang="ts">
  import { ArrowDown, FolderOpen, PanelRight } from "@lucide/svelte";
  import { tick } from "svelte";
  import { app } from "../lib/store.svelte";
  import { shortPath } from "../lib/format";
  import AssistantMessage from "./AssistantMessage.svelte";
  import UserMessage from "./UserMessage.svelte";
  import Composer from "./Composer.svelte";
  import EmptyState from "./EmptyState.svelte";

  let scroller: HTMLDivElement | undefined = $state();
  let atBottom = $state(true);
  const items = $derived(app.session?.transcript ?? []);

  function onScroll() {
    if (!scroller) return;
    atBottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
  }

  function toBottom(smooth = false) {
    scroller?.scrollTo({ top: scroller.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }

  // colle au bas pendant la diffusion, sauf si l'utilisateur remonte lire
  let frame = 0;
  const signature = $derived.by(() => {
    const last = items[items.length - 1];
    if (!last || last.kind === "user") return items.length;
    const b = last.blocks[last.blocks.length - 1];
    const tail = b && "text" in b ? b.text.length : b?.type === "tool" ? b.status.length : 0;
    return `${items.length}:${last.blocks.length}:${tail}:${last.status}:${last.pending_tool ?? ""}`;
  });
  let lastUserCount = 0;
  $effect(() => {
    void signature;
    const users = items.filter((i) => i.kind === "user").length;
    if (users > lastUserCount) atBottom = true;   // nouveau message : on revient toujours en bas
    lastUserCount = users;
    if (!atBottom) return;
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => toBottom());
  });

  $effect(() => {
    void app.session?.id;
    tick().then(() => {
      atBottom = true;
      toBottom();
    });
  });
</script>

<section class="chat">
  {#if app.session}
    <div class="sub">
      <span class="ws" title={app.session.workspace}><FolderOpen size={13} /> {shortPath(app.session.workspace, 60)}</span>
      <button class="icon-btn {app.inspectorOpen ? 'active' : ''}" title="Fichiers (Ctrl B)" onclick={() => (app.inspectorOpen = !app.inspectorOpen)}><PanelRight size={15} /></button>
    </div>
  {/if}

  <div class="scroll" bind:this={scroller} onscroll={onScroll}>
    {#if !items.length}
      <EmptyState />
    {:else}
      <div class="thread">
        {#each items as it, i (it.kind + it.turn_id + i)}
          {#if it.kind === "user"}
            <UserMessage item={it} />
          {:else}
            <AssistantMessage item={it} />
          {/if}
        {/each}
      </div>
    {/if}
  </div>

  {#if !atBottom && items.length}
    <button class="jump fade-in" onclick={() => ((atBottom = true), toBottom(true))} aria-label="Aller en bas"><ArrowDown size={16} /></button>
  {/if}

  <Composer />
</section>

<style>
  .chat { flex: 1; min-height: 0; display: flex; flex-direction: column; position: relative; }
  .sub { display: flex; align-items: center; justify-content: space-between; padding: 8px 14px 0 18px; }
  .ws { display: inline-flex; align-items: center; gap: 7px; font-size: 12px; color: var(--text-3); font-family: var(--mono); }
  .scroll { flex: 1; min-height: 0; overflow-y: auto; overflow-anchor: none; scroll-padding-bottom: 20px; }
  .thread { max-width: 860px; margin: 0 auto; padding: 18px 28px 40px; }
  .jump { position: absolute; bottom: 124px; left: 50%; transform: translateX(-50%); width: 36px; height: 36px; border-radius: 50%; display: grid; place-items: center; background: var(--surface-3); border: 1px solid var(--line-3); box-shadow: var(--shadow-2); color: var(--text); z-index: 3; }
  .jump:hover { background: var(--surface-4); }
</style>
