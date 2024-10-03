ARG SRC_IMAGE="mha-rabbitmq"
ARG SRC_VERSION="latest"
FROM ${SRC_IMAGE}:${SRC_VERSION}
LABEL authors="Dmitry Gnatyshak"

COPY ./mhagenta/ /src/mhagenta/
COPY ./mhagenta/scripts/pyproject.toml /src/pyproject.toml
COPY ./mhagenta/scripts/README.md /src/README.md

RUN python -m pip install -e /src/
