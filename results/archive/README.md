# Archived results

Not part of the current benchmark. Kept for provenance only; do not cite.

## Removed `arithmetic` family

`arith-depth-haiku.json`, `arith-depth-sonnet5.json`, `haiku/`, `sonnet_mono/`,
`reference.json`, `reference.png`, `empty.json`, `contrast-sonnet5.json`.

The benchmark narrowed to information-flow security. A Peano-arithmetic ladder
measures general proving ability, not IFC verification, and it is heavily
represented in model training data, which is the confound the corpus is meant to
avoid. `contrast-sonnet5.json` additionally carried no information: 24 of 24
attempts succeeded, so both arms of the contrast were at ceiling.

## Broken run

`vsi-depth-5.json`. 420 attempts, 0 successes, including 210 compositional
depth-1 tasks that pass in every other run of the same tasks. A pass rate of
exactly zero on tasks known to be solvable indicates a harness fault, not a
policy result. Retained so the failure is on record rather than silently
deleted.
