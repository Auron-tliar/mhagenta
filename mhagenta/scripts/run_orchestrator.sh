#!/bin/sh

if [ "$PRE_VERSION" = "true" ] ; then pip install --pre mhagenta ; else pip install mhagenta ; fi
pip install -r /mha-save/_req/requirements.txt
cd /mha-save/src || exit

python /mha-save/src/orchestrator_launcher.py
