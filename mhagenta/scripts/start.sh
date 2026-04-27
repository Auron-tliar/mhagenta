#!/bin/sh

rabbitmq-server -detached

rabbitmqctl await_startup

python /agent/agent_launcher.py

rabbitmqctl shutdown
