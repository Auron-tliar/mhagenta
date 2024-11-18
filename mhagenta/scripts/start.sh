#!/bin/sh

rabbitmq-server -detached

python /agent/agent_launcher.py

rabbitmqctl shutdown
