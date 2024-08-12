ARG SRC_IMAGE="ha-rabbitmq"
ARG SRC_VERSION="latest"
FROM ${SRC_IMAGE}:${SRC_VERSION}
LABEL authors="Dmitry Gnatyshak"

COPY HAgents/ /src/HAgents/
COPY HAgents/scripts/setup.py /src/setup.py

RUN python -m pip install -e /src/
