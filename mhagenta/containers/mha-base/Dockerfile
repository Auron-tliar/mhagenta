ARG SRC_IMAGE="aurontliar/mhagenta"
ARG SRC_TAG="rmq"
FROM ${SRC_IMAGE}:${SRC_TAG}
LABEL authors="Dmitry Gnatyshak"
ARG PRE_VERSION="false"
ARG LOCAL="false"
ARG TORCH="false"

COPY ./mha-local /mha-local
RUN if [ "$PRE_VERSION" = "true" ] ; then \
        pip install --no-cache-dir --pre mhagenta ;  \
    else \
        if [ "$LOCAL" = "true" ] ; then  \
            cd /mha-local && \
            pip install --no-cache-dir . && \
            cd / ; \
        else \
            pip install --no-cache-dir --upgrade mhagenta ; \
        fi ; \
    fi

RUN if [ "$TORCH" = "true" ] ; then \
        pip install --no-cache-dir torch ;  \
    fi

RUN pip show mhagenta
