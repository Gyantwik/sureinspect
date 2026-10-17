#!/usr/bin/env bash
# Full tracked HEAD tree, not only added/modified paths: renames cannot bypass it.
set -euo pipefail
MAX=5242880; BAD=0
while IFS= read -r -d '' f; do
  case "${f,,}" in
    *.onnx|*.onnx.data|*.npy|*.npz|*.pt|*.pth|*.tar|*.tar.gz|*.tgz|*.zip)
      echo "Forbidden model/archive: $f"; BAD=1 ;;
    .env|.env.*|*/.env|*/.env.*)
      case "$f" in .env.example|*/.env.example) ;; *) echo "Forbidden env file: $f"; BAD=1 ;; esac ;;
  esac
  SIZE=$(git cat-file -s "HEAD:$f")
  if (( SIZE > MAX )); then
    echo "Over 5 MiB: $f ($SIZE bytes)"; BAD=1
  fi
done < <(git ls-tree -r --name-only -z HEAD)
exit "$BAD"
