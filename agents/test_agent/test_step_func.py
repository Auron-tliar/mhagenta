from typing import Any
from pathlib import Path
import logging
import asyncio

from mhagenta import State, Observation
from mhagenta.base import LLReasonerBase, ActuatorBase, PerceptorBase, LearnerBase
from mhagenta.base import KnowledgeBase, HLReasonerBase, GoalGraphBase, MemoryBase
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes import MHARoot
from mhagenta.core import RabbitMQConnector


class TestLLReasoner(LLReasonerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestActuator(ActuatorBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestPerceptor(PerceptorBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestLearner(LearnerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestKnowledge(KnowledgeBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestHLReasoner(HLReasonerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestGoalGraph(GoalGraphBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


class TestMemory(MemoryBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state


async def run_agent(agent: MHARoot) -> None:
    print('INITIALIZING...')
    await agent.initialize()
    print('INITIALIZED!\nRUNNING...')
    await agent.start()
    print('FINISHED!')


if __name__ == '__main__':
    test_agent = MHARoot(
        agent_id='test_agent',
        connector_cls=RabbitMQConnector,
        perceptors=TestPerceptor(
            module_id='test_perceptor',
            initial_state={'step_counter': 0}
        ),
        actuators=TestActuator(
            module_id='test_actuator',
            initial_state={'step_counter': 0}
        ),
        ll_reasoners=TestLLReasoner(
            module_id='test_ll_reasoner',
            initial_state={'step_counter': 0}
        ),
        learners=TestLearner(
            module_id='test_learner',
            initial_state={'step_counter': 0}
        ),
        knowledge=TestKnowledge(
            module_id='test_knowledge',
            initial_state={'step_counter': 0}
        ),
        hl_reasoner=TestHLReasoner(
            module_id='test_hl_reasoner',
            initial_state={'step_counter': 0}
        ),
        goal_graphs=TestGoalGraph(
            module_id='test_goal_graph',
            initial_state={'step_counter': 0}
        ),
        memory=TestMemory(
            module_id='test_memory',
            initial_state={'step_counter': 0}
        ),
        connector_kwargs={
            'host': 'localhost',
            'port': 5672,
            'prefetch_count': 1
        },
        step_frequency=.5,
        exec_duration=10.,
        save_dir=Path('D:\\bsc-tmp\\phd\\phd_thesis\\hybrid-agent\\out\\test_agent_refactor\\basic\\'),
        save_format='json',
        resume=False,
        log_level=logging.DEBUG
    )

    asyncio.run(run_agent(test_agent))
