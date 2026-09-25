#!/bin/sh

# Environment container entry point, installed as /agent/start.sh.
# Requires curl, Python with MHAgentA installed, /agent/environment_launcher.py,
# and /agent/env_params. RMQ_HOST and RMQ_PORT identify the broker's HTTP management
# endpoint, not its AMQP endpoint. The orchestrator supplies the external broker
# host and AMQP port + 10000 (normally 15672), plus DOCKER_NAME and VERBOSE.
# AGENT_ID optionally overrides the environment ID; /out is the host-mounted output.
# Replace the shell with Python so termination signals reach the environment runtime.
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

exec python /agent/environment_launcher.py
