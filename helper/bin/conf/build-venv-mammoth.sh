#! /usr/bin/bash

: "${ACCOUNT:?❌ ACCOUNT is not set}"           # project_462000964
: "${PROJHOME:?❌ PROJHOME is not set}"         # /project/$ACCOUNT/members/$USER

source $PROJHOME/bin/module-loads.sh

: "${MAMMOTH_BASE:?❌ MAMMOTH_BASE is not set}" # $PROJHOME/venv/mammoth
: "${CODE_DIR:?❌ CODE_DIR is not set}"         # $MAMMOTH_BASE/lib64/python3.10/site-packages/mammoth/bin

python -m venv --system-site-packages "$MAMMOTH_BASE"
source $MAMMOTH_BASE/bin/activate
pip install mammoth-nlp
deactivate
ls $CODE_DIR
