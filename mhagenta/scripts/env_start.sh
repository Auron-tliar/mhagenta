#!/bin/sh

if ! curl -fsS --connect-timeout 2 --max-time 3 "http://$RMQ_HOST:$RMQ_PORT/" >/dev/null 2>&1; then
  if [ "$VERBOSE" = "true" ] ; then
    echo "[$DOCKER_NAME] Waiting for RabbitMQ server at $RMQ_HOST:$RMQ_PORT..."
  fi
  until curl -fsS --connect-timeout 2 --max-time 3 "http://$RMQ_HOST:$RMQ_PORT/" >/dev/null 2>&1; do
    sleep 0.5
  done
  if [ "$VERBOSE" = "true" ] ; then
    echo "[$DOCKER_NAME]: RabbitMQ is ready"
  fi
fi

python /agent/environment_launcher.py
