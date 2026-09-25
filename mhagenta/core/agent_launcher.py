"""Container entry point for a serialized agent definition.

The orchestrator installs this script as ``/agent/agent_launcher.py`` and writes a dill parameter dictionary to
``/agent/agent_params``. ``scripts/start.sh`` starts the container's internal RabbitMQ broker before invoking it.
``AGENT_ID`` optionally overrides the serialized agent ID (used for agent copies). ``DOCKER_NAME`` identifies
launcher output and ``VERBOSE`` must be set; the string ``'true'`` enables the version banner. Output paths are
provided in the parameter dictionary, normally with ``/out`` mounted to the host's entity output directory.

This is a container launcher, not the entry point for a user experiment. Define agents through ``Orchestrator``
and call ``run()`` or ``arun()`` from the experiment's own script.
"""

import os
import dill
import asyncio
from typing import Any
import importlib.metadata

from mhagenta.core.processes.mha_root import MHARoot
from mhagenta.modules import *
from mhagenta.states import *
from mhagenta.utils import ModuleTypes, Observation, ActionStatus
from mhagenta.core.processes import run_agent_module, GlobalParams
from mhagenta.core.processes.mha_module import MHAModule, ModuleBase
from mhagenta.bases import *


async def main():
    """Load the container's agent parameters, apply its ID override, and await initialization and execution."""
    with open('/agent/agent_params', 'rb') as f:
        params: dict[str, Any] = dill.load(f)

    id_override = os.environ.get('AGENT_ID')
    if id_override is not None and id_override != '':
        params['agent_id'] = os.environ.get('AGENT_ID')
    agent = MHARoot(**params)
    await agent.initialize()
    await agent.start()

    print(f'[{os.environ['DOCKER_NAME']}] Agent "{agent.agent_id}": execution finished.')


if __name__ == '__main__':
    if os.environ['VERBOSE'] == 'true':
        print(f'[{os.environ['DOCKER_NAME']}] Using MHAgentA version {importlib.metadata.version("mhagenta")}')
    asyncio.run(main())
