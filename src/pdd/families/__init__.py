"""Available ladder families."""

from ..ladder import Family
from .arith import ARITHMETIC
from .noninterference import NONINTERFERENCE

FAMILIES: dict[str, Family] = {f.name: f for f in (ARITHMETIC, NONINTERFERENCE)}

DEFAULT = NONINTERFERENCE.name

__all__ = ["FAMILIES", "DEFAULT", "ARITHMETIC", "NONINTERFERENCE"]
