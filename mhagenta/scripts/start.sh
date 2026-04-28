#!/bin/sh

rabbitmq-server -detached

until rabbitmq-diagnostics -q check_running; do
  sleep 0.5
done

python /agent/agent_launcher.py

rabbitmqctl shutdown
