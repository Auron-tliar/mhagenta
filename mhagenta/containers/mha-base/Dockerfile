ARG SRC_IMAGE="mha-rabbitmq"
ARG SRC_TAG="latest"
FROM ${SRC_IMAGE}:${SRC_TAG}
LABEL authors="Dmitry Gnatyshak"
ARG PRE_VERSION="false"
ARG LOCAL="false"

COPY ./mha-local /mha-local
RUN if [ "$PRE_VERSION" = "true" ] ; then \
        pip install --pre mhagenta ;  \
    else \
        if [ "$LOCAL" = "true" ] ; then  \
            cd /mha-local && \
            pip install . && \
            cd / ; \
        else \
            pip install mhagenta ; \
        fi ; \
    fi
