#!/bin/sh

rabbitmq-server -detached
sleep 10

python /agent/agent_launcher.py

rabbitmqctl shutdown
