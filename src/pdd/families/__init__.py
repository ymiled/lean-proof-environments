"""Available ladder families.

The benchmark covers information-flow security only. `vsi` is the evidence
source: the lemma chain of a machine-checked Volpano-Smith-Irvine development,
absent from public corpora. `noninterference` is the synthetic control: a toy
IFC theory that supports per-seed renaming, which an imported corpus cannot,
and so isolates the contamination variable.

A Peano-arithmetic family was removed. It measured general proving ability
rather than IFC verification, and it is heavily represented in training data,
which is the confound this corpus exists to avoid. Its results are in
`results/archive/`.
"""

from ..ladder import Family
from .noninterference import NONINTERFERENCE
from .vsi import VSI

FAMILIES: dict[str, Family] = {f.name: f for f in (NONINTERFERENCE, VSI)}

DEFAULT = VSI.name

__all__ = ["FAMILIES", "DEFAULT", "NONINTERFERENCE", "VSI"]
