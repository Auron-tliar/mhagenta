import dill
import asyncio
from mhagenta import HAgent
from mhagenta.params import *


async def main():
    with open('/agent/agent_params', 'rb') as f:
        params = dill.load(f)

    params['perceptors'] = [PerceptorParams(**perceptor) for perceptor in params['perceptors']]
    params['actuators'] = [ActuatorParams(**actuator) for actuator in params['actuators']]
    params['ll_reasoner'] = LLParams(**params['ll_reasoner'])
    params['learner'] = LearnerParams(**params['learner']) if params['learner'] is not None else None
    params['memory'] = MemoryParams(**params['memory']) if params['memory'] is not None else None
    params['knowledge'] = KnowledgeParams(**params['knowledge']) if params['knowledge'] is not None else None
    params['hl_reasoner'] = HLParams(**params['hl_reasoner']) if params['hl_reasoner'] is not None else None
    params['goal_graph'] = GGParams(**params['goal_graph']) if params['goal_graph'] is not None else None

    agent = HAgent(**params)
    await agent.initialize_connection()
    async with asyncio.TaskGroup() as tg:
        tg.create_task(agent.start_connector())
        await tg.create_task(agent.initialize_modules())
        await tg.create_task(agent.start())


if __name__ == '__main__':
    asyncio.run(main())
