#!/bin/sh

# Agent container entry point, installed as /agent/start.sh by the orchestrator.
# Requires RabbitMQ tools, Python with MHAgentA installed, /agent/agent_launcher.py,
# and /agent/agent_params. VERBOSE and DOCKER_NAME control launcher output;
# AGENT_ID can override the serialized ID. The orchestrator supplies these values
# and mounts the entity's output directory at /out.
# Wait for the internal AMQP broker, run the agent, then shut the broker down.
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
