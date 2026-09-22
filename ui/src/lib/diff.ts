export interface DiffRow {
  kind: "add" | "del" | "ctx" | "hunk";
  text: string;
  oldNo?: number;
  newNo?: number;
}

/** Diff unifie -> lignes numerotees (en-tetes ---/+++ ignores). */
export function parseUnifiedDiff(src: string): { rows: DiffRow[]; added: number; removed: number } {
  const rows: DiffRow[] = [];
  let o = 0;
  let n = 0;
  let added = 0;
  let removed = 0;
  for (const line of src.split("\n")) {
    if (line.startsWith("--- ") || line.startsWith("+++ ")) continue;
    const h = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$/.exec(line);
    if (h) {
      o = +h[1];
      n = +h[2];
      rows.push({ kind: "hunk", text: line });
      continue;
    }
    if (line.startsWith("+")) {
      rows.push({ kind: "add", text: line.slice(1), newNo: n++ });
      added++;
    } else if (line.startsWith("-")) {
      rows.push({ kind: "del", text: line.slice(1), oldNo: o++ });
      removed++;
    } else if (line === "...") {
      rows.push({ kind: "hunk", text: "…" });
    } else if (rows.length) {
      rows.push({ kind: "ctx", text: line.startsWith(" ") ? line.slice(1) : line, oldNo: o++, newNo: n++ });
    }
  }
  return { rows, added, removed };
}

/** Deux extraits (edit_file en attente d'autorisation) -> diff minimal ligne a ligne. */
export function simpleDiff(oldText: string, newText: string): DiffRow[] {
  const a = oldText.split("\n");
  const b = newText.split("\n");
  const rows: DiffRow[] = [];
  let i = 0;
  while (i < a.length && i < b.length && a[i] === b[i]) rows.push({ kind: "ctx", text: a[i], oldNo: i + 1, newNo: ++i });
  let ea = a.length - 1;
  let eb = b.length - 1;
  const tail: DiffRow[] = [];
  while (ea >= i && eb >= i && a[ea] === b[eb]) tail.unshift({ kind: "ctx", text: a[ea], oldNo: ea + 1, newNo: eb + 1 }), ea--, eb--;
  for (let k = i; k <= ea; k++) rows.push({ kind: "del", text: a[k], oldNo: k + 1 });
  for (let k = i; k <= eb; k++) rows.push({ kind: "add", text: b[k], newNo: k + 1 });
  return rows.concat(tail);
}
