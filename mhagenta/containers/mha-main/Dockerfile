ARG SRC_IMAGE="mha-base"
ARG SRC_VERSION="latest"
FROM ${SRC_IMAGE}:${SRC_VERSION}
LABEL authors="Dmitry Gnatyshak"

COPY src /agent/
RUN mkdir -p /out/logs
RUN mkdir -p /out/save
RUN if [ -e /agent/init_script.sh ] ; then sh /agent/init_script.sh ; fi

RUN pip install -r "/agent/requirements.txt"

ENTRYPOINT ["sh"]
CMD ["/agent/start.sh"]
