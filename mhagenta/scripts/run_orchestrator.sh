#!/bin/sh

PYTHONUNBUFFERED=1
apk add --update --no-cache python3 py3-pip
apk add python3-tkinter
python -m venv /venv/
. /venv/bin/activate

if [ "$PRE_VERSION" = "true" ] ; then pip install --pre mhagenta ; else pip install mhagenta ; fi
pip install -r /mha-save/_req/requirements.txt
cd /mha-save/src || exit

python /mha-save/src/orchestrator_launcher.py
