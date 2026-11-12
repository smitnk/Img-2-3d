#!/usr/bin/env bash
# Download an artifact of the newest successful run (with a not yet expired copy) of another workflow.
#   GH_TOKEN=... GITHUB_REPOSITORY=owner/repo fetch_artifact.sh <workflow file> <artifact name> <dest dir> [run id]
set -euo pipefail
wf=$1; art=$2; dest=$3; run=${4:-}
if [ -z "$run" ]; then
  for id in $(gh api "repos/$GITHUB_REPOSITORY/actions/workflows/$wf/runs?status=success&per_page=20" --jq '.workflow_runs[].id'); do
    n=$(gh api "repos/$GITHUB_REPOSITORY/actions/runs/$id/artifacts" --jq "[.artifacts[] | select(.name == \"$art\" and .expired == false)] | length")
    if [ "$n" -gt 0 ]; then run=$id; break; fi
  done
fi
[ -n "$run" ] || { echo "::error::no successful $wf run with a live '$art' artifact (they expire: skel after 1 day, skin after 7) - re-run that workflow" >&2; exit 1; }
echo "$wf -> run $run, artifact $art -> $dest"
gh run download "$run" -R "$GITHUB_REPOSITORY" -n "$art" -D "$dest"
