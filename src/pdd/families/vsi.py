"""The benchmark ladder extracted from the full Lean Vsi development.

Unlike the older ``noninterference`` family, this module does not restate a
toy language.  It reads the checked declarations from ``lean/Vsi.lean`` and
uses their actual reference proof blocks as rungs.  Names remain canonical in
this family because the source itself is the object being benchmarked.
"""

from __future__ import annotations

from pathlib import Path

from ..ladder import Family, Rung


_SOURCE = (Path(__file__).resolve().parents[3] / "lean" / "Vsi.lean").read_text()
_MARK = "theorem lowEq_refl"
_prefix, _rest = _SOURCE.split(_MARK, 1)

# Definitions introduced after the initial theorem block and needed by the
# full proof.  They are declarations, not theorem answers, so they stay in the
# context for every task.
def _between(text: str, start: str, stop: str) -> str:
    return text.split(start, 1)[1].split(stop, 1)[0]


_defs = [_prefix]
_defs.append("def secure" + _between(_SOURCE, "def secure", "theorem secure_sub"))
_defs.append("def noCall" + _between(_SOURCE, "def noCall", "/-- Public call-free"))
_defs.append("def noCallC" + _between(_SOURCE, "def noCallC", "/-- **Noninterference."))
_defs.append("structure FTOk" + _between(_SOURCE, "structure FTOk", "theorem agree"))
DEFINITIONS = "\n\n".join(_defs)


def _raw(name: str) -> str:
    start = _SOURCE.index("theorem " + name)
    markers = (
        "\n/--", "\ntheorem ", "\ndef ", "\nstructure ",
        "\n#print axioms",
    )
    ends = [p for marker in markers
            if (p := _SOURCE.find(marker, start + 1)) >= 0]
    end = min(ends) if ends else len(_SOURCE)
    raw = _SOURCE[start:end].strip()
    if ":= by" not in raw:
        head, body = raw.split(":=", 1)
        raw = head.rstrip() + ":= by\n  exact " + body.strip()
    return raw


_deps = {
    "lowEq_refl": (),
    "lowEq_symm": ("lowEq_refl",),
    "lowEq_trans": ("lowEq_refl", "lowEq_symm"),
    "Lvl.le_refl": (),
    "Lvl.le_trans": ("Lvl.le_refl",),
    "sub_base_inv": ("Lvl.le_refl", "Lvl.le_trans"),
    "tyE_var_L": ("sub_base_inv",),
    "tyE_binop_L": ("sub_base_inv",),
    "evalE_agree_nocall": ("lowEq_refl", "tyE_var_L", "tyE_binop_L"),
    "secure_sub": ("Lvl.le_refl", "Lvl.le_trans"),
    "confinement": ("lowEq_refl", "lowEq_trans", "secure_sub"),
    "tyE_call_L": ("sub_base_inv",),
    "agree": (
        "lowEq_refl", "lowEq_symm", "lowEq_trans", "tyE_var_L",
        "tyE_binop_L", "tyE_call_L", "confinement", "secure_sub",
    ),
    "noninterference_full": ("agree",),
}

RUNGS = tuple(
    Rung(
        key=key,
        role="machine-checked Vsi proof obligation",
        binders="",
        statement="True",
        proof="exact True.intro",
        deps=deps,
        raw=_raw(key),
    )
    for key, deps in _deps.items()
)

VSI = Family(name="vsi", definitions=DEFINITIONS, rungs=RUNGS, slots={})
