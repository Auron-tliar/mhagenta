from pathlib import Path
from .CONTAINER_VERSION import CONTAINER_VERSION


REPO = 'aurontliar/mhagenta'
RABBIT_IMG_PATH = str((Path(__file__).parent / 'mha-rabbitmq/').resolve())
BASE_IMG_PATH = str((Path(__file__).parent / 'mha-base/').resolve())
AGENT_IMG_PATH = str((Path(__file__).parent / 'mha-main/').resolve())


__all__ = ['REPO', 'RABBIT_IMG_PATH', 'BASE_IMG_PATH', 'AGENT_IMG_PATH', 'CONTAINER_VERSION']
