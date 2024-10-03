import dill
import asyncio
from mhagenta.core.processes import MHARoot


async def main():
    with open('/agent/agent_params', 'rb') as f:
        params = dill.load(f)

    agent = MHARoot(**params)
    await agent.initialize()
    await agent.start()


if __name__ == '__main__':
    asyncio.run(main())
