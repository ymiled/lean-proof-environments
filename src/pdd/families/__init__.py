"""Available ladder families."""

from ..ladder import Family
from .arith import ARITHMETIC
from .noninterference import NONINTERFERENCE
from .vsi import VSI

FAMILIES: dict[str, Family] = {f.name: f for f in (ARITHMETIC, NONINTERFERENCE, VSI)}

DEFAULT = NONINTERFERENCE.name

__all__ = ["FAMILIES", "DEFAULT", "ARITHMETIC", "NONINTERFERENCE", "VSI"]
