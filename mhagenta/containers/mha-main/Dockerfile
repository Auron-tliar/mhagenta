ARG SRC_IMAGE="mha-base"
ARG SRC_VERSION="latest"
FROM ${SRC_IMAGE}:${SRC_VERSION}
LABEL authors="Dmitry Gnatyshak"

COPY src /agent/
ENV PYTHONPATH="/agent:${PYTHONPATH}"
RUN mkdir -p /out/logs
RUN mkdir -p /out/save
RUN if [ -e /agent/init_script.sh ] ; then sh /agent/init_script.sh ; fi

RUN if [ -e /agent/requirements.txt ] ; then pip install -r "/agent/requirements.txt" ; fi

ENTRYPOINT ["sh"]
CMD ["/agent/start.sh"]
