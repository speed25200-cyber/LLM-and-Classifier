import { Marked, type Tokens } from "marked";
import DOMPurify from "dompurify";
import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import c from "highlight.js/lib/languages/c";
import cpp from "highlight.js/lib/languages/cpp";
import css from "highlight.js/lib/languages/css";
import diff from "highlight.js/lib/languages/diff";
import go from "highlight.js/lib/languages/go";
import java from "highlight.js/lib/languages/java";
import javascript from "highlight.js/lib/languages/javascript";
import json from "highlight.js/lib/languages/json";
import markdown from "highlight.js/lib/languages/markdown";
import powershell from "highlight.js/lib/languages/powershell";
import python from "highlight.js/lib/languages/python";
import rust from "highlight.js/lib/languages/rust";
import sql from "highlight.js/lib/languages/sql";
import typescript from "highlight.js/lib/languages/typescript";
import xml from "highlight.js/lib/languages/xml";
import yaml from "highlight.js/lib/languages/yaml";

const langs: Record<string, any> = { bash, c, cpp, css, diff, go, java, javascript, json, markdown, powershell, python, rust, sql, typescript, xml, yaml };
for (const [name, def] of Object.entries(langs)) hljs.registerLanguage(name, def);
const alias: Record<string, string> = { js: "javascript", ts: "typescript", py: "python", sh: "bash", shell: "bash", zsh: "bash", ps1: "powershell",
  html: "xml", svg: "xml", yml: "yaml", md: "markdown", rs: "rust", tsx: "typescript", jsx: "javascript", svelte: "xml", vue: "xml", console: "bash" };

const esc = (s: string) => s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]!);

export function highlight(code: string, lang?: string): string {
  const l = lang ? (alias[lang.toLowerCase()] ?? lang.toLowerCase()) : "";
  try {
    if (l && hljs.getLanguage(l)) return hljs.highlight(code, { language: l, ignoreIllegals: true }).value;
    if (code.length < 20_000) return hljs.highlightAuto(code, ["python", "javascript", "typescript", "bash", "json", "xml", "css"]).value;
  } catch {
    /* coloration impossible : texte brut */
  }
  return esc(code);
}

export function langFromPath(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  return alias[ext] ?? ext;
}

const md = new Marked({
  gfm: true,
  breaks: false,
  renderer: {
    code(token: Tokens.Code) {
      const lang = (token.lang ?? "").split(/\s/)[0];
      return `<div class="md-code"><div class="md-code-head"><span>${esc(lang || "code")}</span><button type="button" class="md-copy" data-copy>Copier</button></div><pre><code class="hljs">${highlight(token.text, lang)}</code></pre></div>`;
    },
    link(token: Tokens.Link) {
      const text = this.parser.parseInline(token.tokens);
      return `<a href="${esc(token.href)}" target="_blank" rel="noopener noreferrer">${text}</a>`;
    },
  },
});

DOMPurify.addHook("afterSanitizeAttributes", (node) => {
  if (node.tagName === "A") node.setAttribute("rel", "noopener noreferrer");
});

export function renderMarkdown(src: string): string {
  const html = md.parse(src, { async: false }) as string;
  return DOMPurify.sanitize(html, { ADD_ATTR: ["target", "data-copy"], FORBID_TAGS: ["style", "form", "input"] });
}

/** Clic delegue : boutons "Copier" des blocs de code et liens externes (ouverts hors de l'application). */
export function handleMarkdownClick(e: MouseEvent) {
  const t = e.target as HTMLElement;
  const btn = t.closest("[data-copy]") as HTMLElement | null;
  if (btn) {
    const code = btn.closest(".md-code")?.querySelector("code")?.textContent ?? "";
    navigator.clipboard?.writeText(code);
    btn.textContent = "Copie ✓";
    setTimeout(() => (btn.textContent = "Copier"), 1400);
    return;
  }
  const a = t.closest("a[href]") as HTMLAnchorElement | null;
  if (a && /^https?:/.test(a.href)) {
    e.preventDefault();
    const opener = window.__TAURI__?.opener;
    if (opener?.openUrl) opener.openUrl(a.href);
    else window.open(a.href, "_blank", "noopener");
  }
}
