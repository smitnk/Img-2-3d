#!/bin/bash
# push_out.sh VARIANT SRC_DIR  -> adds VARIANT/ to branch paint-test-out (non-force, retry; parallel-job safe)
V=$1; SRC=$2; URL="https://x-access-token:${GH_TOKEN}@github.com/${REPO}"
for i in 1 2 3 4 5 6 7 8; do
  rm -rf /tmp/o
  if git ls-remote --exit-code "$URL" paint-test-out >/dev/null 2>&1; then git clone -q --depth 1 -b paint-test-out "$URL" /tmp/o
  else mkdir /tmp/o && git -C /tmp/o init -q -b paint-test-out && git -C /tmp/o remote add origin "$URL"; fi
  rm -rf "/tmp/o/$V"; mkdir -p "/tmp/o/$V"; cp -r "$SRC"/. "/tmp/o/$V/" 2>/dev/null
  rm -f "/tmp/o/$V/latents.pt"
  cd /tmp/o && git add -A && git -c user.name=ci -c user.email=ci@x commit -qm "out $V" && git push -q origin HEAD:paint-test-out && exit 0
  cd /; sleep $((RANDOM % 15 + 3))
done
exit 1
