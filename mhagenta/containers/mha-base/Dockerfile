ARG SRC_IMAGE="mha-rabbitmq"
ARG SRC_VERSION="latest"
FROM ${SRC_IMAGE}:${SRC_VERSION}
LABEL authors="Dmitry Gnatyshak"
ARG PRE_VERSION="false"

RUN if [ "$PRE_VERSION" = "true" ] ; then pip install --pre mhagenta ; else pip install mhagenta ; fi
