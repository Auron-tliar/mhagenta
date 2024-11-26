ARG SRC_IMAGE="mha-rabbitmq"
ARG SRC_TAG="latest"
FROM ${SRC_IMAGE}:${SRC_TAG}
LABEL authors="Dmitry Gnatyshak"
ARG PRE_VERSION="false"

RUN if [ "$PRE_VERSION" = "true" ] ; then pip install --pre mhagenta ; else pip install mhagenta ; fi
