#!/bin/bash
# Build the FluxFill dashboard locally and publish it to deepearth.ai/fluxfill as a new release.
# The server only serves files: the folder is built here and copied; the `site` symlink switches atomically.
# Usage: FLUXFILL_DEPLOY_HOST=user@host FLUXFILL_DEPLOY_KEY=~/.ssh/key FLUXFILL_DEPLOY_DIR=/srv/fluxfill deploy.sh
set -euo pipefail
HOST=${FLUXFILL_DEPLOY_HOST:?set FLUXFILL_DEPLOY_HOST}
KEY="-i ${FLUXFILL_DEPLOY_KEY:?set FLUXFILL_DEPLOY_KEY}"
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${FLUXFILL_BUILD:-/home/photon/4tb/fluxfill/dashboard_build/site}
REMOTE=${FLUXFILL_DEPLOY_DIR:?set FLUXFILL_DEPLOY_DIR}
R=$(date +%Y%m%d%H%M%S)
python3 -I "$HERE/build.py" "$OUT"
ssh $KEY "$HOST" "mkdir -p $REMOTE/releases/$R"
rsync -a --chmod=D750,F640 -e "ssh $KEY" "$OUT/" "$HOST:$REMOTE/releases/$R/"
ssh $KEY "$HOST" "cd $REMOTE && chgrp -R www-data releases/$R && ln -sfn $REMOTE/releases/$R site.tmp && mv -T site.tmp site \
  && chgrp -h www-data site && ls -1d releases/* | head -n -5 | xargs -r rm -rf"
curl -s -o /dev/null -w "deepearth.ai/fluxfill/ %{http_code}\n" https://deepearth.ai/fluxfill/
