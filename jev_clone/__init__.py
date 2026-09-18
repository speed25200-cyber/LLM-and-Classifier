"""jev_clone: un modele de decision "System One" (clone ouvert de Jev / TypeSafe AI)
qui lit des probabilites calibrees sur des options typees, en une passe, sans generer de texte.

Il tourne sur n'importe quel GGUF servi par llama-server (fork PrismML ou mainline), y compris
Bonsai 2 27B lui-meme ("mode mono") ou un petit modele dedie (Ternary-Bonsai-1.7B, Qwen3.5-0.8B...).
"""

from jev_clone.schema import SystemOneRequest, SystemOneResponse
from jev_clone.engine import SystemOneEngine
from jev_clone.readout import Calibration

__all__ = ["SystemOneRequest", "SystemOneResponse", "SystemOneEngine", "Calibration"]
__version__ = "0.1.0"
