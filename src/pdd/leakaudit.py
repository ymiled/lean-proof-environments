"""Post-hoc leak audit: did a response reproduce the reference proof verbatim?

Monolithic prompts contain no proofs at all, so a candidate block that matches
the reference proof closely did not come from the prompt. It came from the
repository. This catches the failure mode that corrupted an earlier sweep, where
agents transcribed proofs they had been told not to open.

Similarity is on normalised tactic tokens, so whitespace and layout differences
do not mask a copy and do not manufacture one either.

Length matters more than the ratio. Several rungs here have a reference proof of
under twenty tokens whose form is effectively forced by the goal -- `intro`,
`cases`, `simp` -- and an honest attempt reproduces it exactly. Flagging those
would report leakage on every run and mean nothing. So the audit reports the
ratio against reference length and only treats a match as evidence when the
reference is long enough that independent convergence is implausible.
"""
import difflib, json, re, sys
from pathlib import Path
from .families import FAMILIES
from .ladder import Instance
from .task import Task, parse_blocks

#: A reference this long is not reproduced verbatim by chance.
LONG = 40


def norm(s: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_.']*|\S", s)

def main() -> None:
    _run()


def _run():
  specs = json.loads(Path("runs/_specs.json").read_text())
  fam = FAMILIES[specs[0]["family"]]
  ids = set(sys.argv[1:]) or None
  worst = []
  for s in specs:
      if ids and s["task_id"] not in ids: continue
      out = Path("runs") / f"{s['task_id']}.out"
      if not out.exists(): continue
      inst = Instance.sample(fam, s["seed"])
      task = Task(inst, tuple(s["targets"]), s["arm"])
      blocks = parse_blocks(out.read_text(), task)
      if isinstance(blocks, str): blocks = {task.targets[0]: blocks}
      for t in task.targets:
          cand = (blocks.get(t) or "").strip()
          if not cand: continue
          ref = inst.reference_proof_of(t).strip()
          r = difflib.SequenceMatcher(None, norm(cand), norm(ref)).ratio()
          worst.append((r, s["task_id"], t, len(norm(ref))))
  worst.sort(reverse=True)
  print(f"{'sim':>6} {'ref toks':>9}  task / target")
  for r, tid, t, n in worst[:18]:
      flag = "  <-- CHECK" if r > 0.85 and n >= LONG else ""
      print(f"{r:>6.2f} {n:>9}  {tid} :: {t}{flag}")
  hi = [w for w in worst if w[0] > 0.85 and w[3] >= LONG]
  print(f"\n{len(hi)} of {len(worst)} blocks match a reference of >= {LONG} tokens at > 0.85")
  lens = sorted(w[3] for w in worst)
  print(f"reference lengths: min {lens[0]}, median {lens[len(lens)//2]}, max {lens[-1]}")
  byl = [(w[0], w[3]) for w in worst if w[3] >= LONG]
  if byl:
      print(f"long-reference blocks: {len(byl)}, max similarity {max(r for r,_ in byl):.2f}")


if __name__ == "__main__":
    main()
