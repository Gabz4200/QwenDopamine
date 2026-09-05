"""General-purpose reward conditioning building blocks — re-export shim.

The classes are split across per-concern modules:

  - :mod:`.scalers` — :class:`AsinhScaler`, :class:`LearnableSoftsign`
  - :mod:`.fourier` — :class:`LearnableFourierFeatures`
  - :mod:`.film`    — :class:`TokenWiseFiLM`
"""

from qwendopamine.models.blocks.reward.film import TokenWiseFiLM
from qwendopamine.models.blocks.reward.fourier import LearnableFourierFeatures
from qwendopamine.models.blocks.reward.scalers import AsinhScaler, LearnableSoftsign

__all__ = [
    "AsinhScaler",
    "LearnableFourierFeatures",
    "LearnableSoftsign",
    "TokenWiseFiLM",
]
