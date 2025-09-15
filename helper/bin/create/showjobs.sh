#!/usr/bin/bash
set -euo pipefail
: "${PROJHOME:?❌ PROJHOME not set}"

ls $PROJHOME/data
# retrospective wisdom: this could be renamed as jobs (or something better)

