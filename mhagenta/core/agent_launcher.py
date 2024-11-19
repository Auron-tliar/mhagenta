import os
import dill
import asyncio
from typing import Any
import importlib.metadata

from mhagenta.core.processes import MHARoot
from mhagenta.modules import *
from mhagenta.states import *
from mhagenta.utils import ModuleTypes, Observation, ActionStatus
from mhagenta.core.processes import run_agent_module, MHAModule, ModuleBase, GlobalParams
from mhagenta.bases import *


async def main():
    with open('/agent/agent_params', 'rb') as f:
        params: dict[str, Any] = dill.load(f)

    id_override = os.environ.get('AGENT_ID')
    if id_override is not None and id_override != '':
        params['agent_id'] = os.environ.get('AGENT_ID')
    agent = MHARoot(**params)
    await agent.initialize()
    await agent.start()


if __name__ == '__main__':
    print(f'Using MHAgentA version {importlib.metadata.version("mhagenta")}')
    asyncio.run(main())
