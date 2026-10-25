#!/usr/bin/env bash
set -euo pipefail
P1="Biswabasini90"; P2="Gyantwik"; P3="Khushbu656"; P4="K842"
AUTHOR="${AUTHOR:?}"; BASE="${BASE:?}"; LABELS=",${LABELS:-},"
case "${AUTHOR,,}" in
  "${P1,,}") OWN='^(core|train|config|data/templates|docs/failure_cases)/' ;;
  "${P2,,}") OWN='^(backend|deploy|eval|data/samples)/|^docs/aws_section\.md$|^core/(degrade|quality_gate)\.py$|^core/tests/test_(degrade|quality_gate)\.py$' ;;
  "${P3,,}") OWN='^(frontend|docs)/' ;;
  "${P4,,}") OWN='^(deploy)/|^docs/aws_section\.md$' ;;
  *) echo "Unknown author: $AUTHOR"; exit 1 ;;
esac
SHARED='^(README\.md|TEAM_SYNC\.md|requirements[^/]*\.txt|\.gitignore|\.gitattributes|\.dockerignore|\.env\.example)$|^\.github/'
CONTRACT='^contracts/|^backend/schemas\.py$|^core/interfaces\.py$'
BAD=0
while IFS= read -r -d '' f; do
  if [[ "$f" =~ $CONTRACT ]]; then
    [[ "$LABELS" == *",contract-ok,"* ]] || { echo "Needs contract-ok: $f"; BAD=1; }
  elif [[ "$f" =~ $SHARED ]]; then
    [[ "$LABELS" == *",shared-ok,"* ]] || { echo "Needs shared-ok: $f"; BAD=1; }
  elif [[ "${AUTHOR,,}" == "${P3,,}" && ( "$f" =~ ^docs/failure_cases/ || "$f" == "docs/aws_section.md" ) ]]; then
    echo "Another owner's path: $f"; BAD=1
  elif ! [[ "$f" =~ $OWN ]]; then
    echo "Outside $AUTHOR ownership: $f"; BAD=1
  fi
done < <(git diff --no-renames --name-only -z "$BASE"...HEAD)
exit "$BAD"
