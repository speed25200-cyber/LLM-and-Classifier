"""Prophet Studio : la plateforme locale de la fusion Jev-clone (System One) x Bonsai 2 27B (System Two).

Un coeur Python (FastAPI) qui detecte le materiel, planifie la VRAM (RTX 5060 8 Go en cible), installe les
modeles et le runtime llama.cpp, supervise les serveurs, fait tourner l'agent Prophet en flux continu, et
porte la voix (reconnaissance, synthese, commandes vocales jugees par le classifieur). L'interface (ui/) et
l'application de bureau (desktop/) se branchent dessus en HTTP + WebSocket sur 127.0.0.1.
"""

__version__ = "0.2.0"
