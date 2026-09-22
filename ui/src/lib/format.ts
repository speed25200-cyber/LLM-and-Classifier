const nf1 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 1 });
const nf0 = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 0 });

export const num = (x: number, digits = 0) => (digits ? nf1 : nf0).format(x);

export function bytes(b: number): string {
  if (!b) return "0 o";
  const u = ["o", "Ko", "Mo", "Go", "To"];
  const i = Math.min(u.length - 1, Math.floor(Math.log(b) / Math.log(1000)));
  return `${nf1.format(b / 1000 ** i)} ${u[i]}`;
}

export const gb = (x: number) => `${nf1.format(x)} Go`;
export const mibToGib = (m: number) => `${nf1.format(m / 1024)} Gio`;

export function ms(x: number | null | undefined): string {
  if (x == null) return "—";
  if (x < 1000) return `${nf0.format(x)} ms`;
  if (x < 60_000) return `${nf1.format(x / 1000)} s`;
  const m = Math.floor(x / 60_000);
  return `${m} min ${nf0.format((x % 60_000) / 1000)} s`;
}

export function eta(s: number | null | undefined): string {
  if (s == null || !isFinite(s)) return "";
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

export const pct = (x: number | null | undefined) => (x == null ? "—" : `${nf0.format(x * 100)} %`);

export function ago(ts: number): string {
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "a l'instant";
  if (d < 3600) return `il y a ${Math.floor(d / 60)} min`;
  if (d < 86400) return `il y a ${Math.floor(d / 3600)} h`;
  const days = Math.floor(d / 86400);
  return days === 1 ? "hier" : `il y a ${days} j`;
}

export const ctxLabel = (c: number) => (c >= 1024 ? `${Math.round(c / 1024)} k` : `${c}`);

export function shortPath(p: string, max = 42): string {
  if (!p || p.length <= max) return p;
  const parts = p.split(/[\\/]/);
  let out = parts.pop() ?? "";
  while (parts.length && out.length < max - 6) out = parts.pop() + "/" + out;
  return "…/" + out;
}
