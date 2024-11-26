import dill
from typing import Any
from pathlib import Path

import mhagenta
from mhagenta import Orchestrator


if __name__ == "__main__":
    with open('/mha-save/_orchestrator', 'rb') as f:
        orchestrator: Orchestrator = dill.load(f)
    with open('/mha-save/_params', 'rb') as f:
        params: dict[str, Any] = dill.load(f)
    orchestrator._save_dir = Path(orchestrator._save_dir)
    orchestrator._package_dir = str(Path(mhagenta.__file__).parent.absolute())
    orchestrator._docker_init()
    orchestrator.run(**params)
