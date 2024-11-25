import dill
from typing import Any

from mhagenta import Orchestrator


if __name__ == "__main__":
    with open('/mha-save/_orchestrator', 'rb') as f:
        orchestrator: Orchestrator = dill.load(f)
    with open('/mha-save/_params', 'rb') as f:
        params: dict[str, Any] = dill.load(f)
    orchestrator.run(**params)
