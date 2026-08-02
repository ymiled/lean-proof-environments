#!/bin/sh
# Build a self-contained extraction wrapper around a standalone Lean file.
# Usage: ./mkextract.sh Vsi.lean vsi-graph.json
set -e
SRC="$1"; OUT="$2"; TMP="_Extract_$(basename "$SRC")"
{ echo "import PddExtract"; cat "$SRC"; \
  echo ""; echo "open Lean Elab Command in"; \
  echo "#eval PddExtract.run \`Local \"$OUT\""; } > "$TMP"
LEAN_PATH="$PWD" lean "$TMP" 2>&1 | grep -v "depends on axioms\|does not depend"
rm -f "$TMP"
