#!/bin/bash
# push_branch.sh SUBDIR SRC_DIR -> adds SUBDIR/ to branch paint-phone-out (non-force, retried; parallel-job safe). Needs GH_TOKEN, REPO.
V=$1; SRC=$2; B=paint-phone-out; URL="https://x-access-token:${GH_TOKEN}@github.com/${REPO}"
for i in 1 2 3 4 5 6 7 8; do
  rm -rf /tmp/o
  if git ls-remote --exit-code "$URL" $B >/dev/null 2>&1; then git clone -q --depth 1 -b $B "$URL" /tmp/o
  else mkdir /tmp/o && git -C /tmp/o init -q -b $B && git -C /tmp/o remote add origin "$URL"; fi
  rm -rf "/tmp/o/$V"; mkdir -p "/tmp/o/$V"; cp -r "$SRC"/. "/tmp/o/$V/" 2>/dev/null
  rm -rf "/tmp/o/$V/.paint_work" "/tmp/o/$V/mesh_uv.npz.tmp"
  cd /tmp/o && git add -A && git -c user.name=ci -c user.email=ci@x commit -qm "out $V" && git push -q origin HEAD:$B && exit 0
  cd /; sleep $((RANDOM % 15 + 3))
done
exit 1
