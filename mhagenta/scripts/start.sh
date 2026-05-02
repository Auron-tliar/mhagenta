#!/bin/sh

rabbitmq-server -detached
echo "Waiting for RabbitMQ..."

until rabbitmq-diagnostics -q check_running >/dev/null 2>&1; do
  sleep 0.5
done

echo "RabbitMQ is ready"
python /agent/agent_launcher.py

rabbitmqctl shutdown
