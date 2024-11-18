ARG SRC_IMAGE="mha-base"
ARG SRC_VERSION="latest"
FROM ${SRC_IMAGE}:${SRC_VERSION}
LABEL authors="Dmitry Gnatyshak"

COPY src /agent/
RUN mkdir -p /out/logs
RUN mkdir -p /out/save

ENTRYPOINT ["sh"]
CMD ["/agent/start.sh"]
