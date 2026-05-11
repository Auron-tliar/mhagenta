#!/bin/sh

rabbitmq-server -detached
if [ "$VERBOSE" = "true" ] ; then
  echo "[$DOCKER_NAME] Waiting for internal RabbitMQ..."
fi

until rabbitmq-diagnostics -q check_running >/dev/null 2>&1; do
  sleep 0.5
done

if [ "$VERBOSE" = "true" ] ; then
  echo "[$DOCKER_NAME] internal RabbitMQ is ready"
fi

python /agent/agent_launcher.py

rabbitmqctl shutdown >/dev/null 2>&1
