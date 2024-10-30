from pathlib import Path
from .CONTAINER_VERSION import CONTAINER_VERSION


RABBIT_IMG_PATH = str((Path(__file__).parent / 'mha-rabbitmq/').absolute())
BASE_IMG_PATH = str((Path(__file__).parent / 'mha-base/').absolute())
AGENT_IMG_PATH = str((Path(__file__).parent / 'mha-main/').absolute())


__all__ = ['RABBIT_IMG_PATH', 'BASE_IMG_PATH', 'AGENT_IMG_PATH', 'CONTAINER_VERSION']
