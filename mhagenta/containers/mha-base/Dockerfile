ARG SRC_IMAGE="aurontliar/mhagenta"
ARG SRC_TAG="rmq"
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
            pip install --upgrade mhagenta ; \
        fi ; \
    fi

RUN pip show mhagenta
