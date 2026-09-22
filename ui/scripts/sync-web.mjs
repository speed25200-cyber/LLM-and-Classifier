// Copie ui/dist -> prophet_studio/web : le paquet Python sert l'interface sans Node chez l'utilisateur.
import { cpSync, existsSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = resolve(here, "../dist");
const web = resolve(here, "../../prophet_studio/web");
if (!existsSync(dist)) throw new Error("ui/dist absent : lancer `npm run build`");
rmSync(web, { recursive: true, force: true });
cpSync(dist, web, { recursive: true });
console.log(`interface copiee dans ${web}`);
