"""Container entry point for a serialized environment definition.

The orchestrator installs this script beneath ``/agent`` with the dill dictionary ``/agent/env_params``. That
dictionary contains the transport runtime class as ``env_class`` and its constructor arguments, including the
behaviour instance. ``AGENT_ID``, when non-empty, overrides ``env_id``; ``DOCKER_NAME`` is required for launcher
output. Save paths come from the parameters, normally targeting the host-mounted ``/out`` directory.

``scripts/env_start.sh`` waits for the external broker's HTTP management endpoint and then invokes this script.
Experiments should register environments through ``Orchestrator.add_environment()`` and launch the orchestrator.
"""

import os
import dill
import asyncio
from typing import Any
from pathlib import Path

from mhagenta.environment import MHAEnvironment


async def main() -> None:
    """Load the container's environment parameters, apply its ID override, and await initialization and execution."""
    with open(Path('/agent/env_params').as_posix(), 'rb') as f:
        params: dict[str, Any] = dill.load(f)

    id_override = os.environ.get('AGENT_ID')
    if id_override is not None and id_override != '':
        params['env_id'] = os.environ.get('AGENT_ID')

    env_class: type[MHAEnvironment] = params.pop('env_class')
    env = env_class(**params)

    await env.initialize()
    await env.start()

    print(f'[{os.environ['DOCKER_NAME']}] Environment "{env.id}": execution finished.')


if __name__ == '__main__':
    asyncio.run(main())
