<script lang="ts">
  import { handleMarkdownClick, renderMarkdown } from "../lib/markdown";

  // Pendant la diffusion, le rendu est limite a ~20 images/s : fluide sans recalculer le markdown a chaque token.
  let { text, streaming = false }: { text: string; streaming?: boolean } = $props();
  let html = $state("");
  let latest = "";
  let scheduled = false;

  $effect(() => {
    latest = text;
    if (!streaming) {
      html = renderMarkdown(latest);
      return;
    }
    if (!scheduled) {
      scheduled = true;
      setTimeout(() => {
        scheduled = false;
        html = renderMarkdown(latest);
      }, 50);
    }
  });
</script>

<!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
<div class="md" class:streaming onclick={handleMarkdownClick}>{@html html}</div>

<style>
  .streaming :global(> :last-child::after) {
    content: ""; display: inline-block; width: 7px; height: 1.05em; margin-left: 2px; vertical-align: -0.15em;
    border-radius: 2px; background: var(--s2); animation: caret 1s steps(2) infinite;
  }
  @keyframes caret { 50% { opacity: 0; } }
</style>
