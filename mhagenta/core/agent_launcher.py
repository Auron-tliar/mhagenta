import dill
import asyncio

from mhagenta.core.processes import MHARoot
from mhagenta.modules import *
from mhagenta.states import *
from mhagenta.utils import ModuleTypes, Observation, ActionStatus
from mhagenta.core.processes import run_agent_module, MHAModule, ModuleBase, GlobalParams
from mhagenta.base import *


async def main():
    with open('/agent/agent_params', 'rb') as f:
        params = dill.load(f)

    agent = MHARoot(**params)
    await agent.initialize()
    await agent.start()


if __name__ == '__main__':
    asyncio.run(main())
