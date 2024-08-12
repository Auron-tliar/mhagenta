from typing import Any
from pathlib import Path
import logging
import asyncio

from mhagenta import State, Observation
from mhagenta.base import LLReasonerBase, ActuatorBase, PerceptorBase
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes import MHARoot
from mhagenta.core import RabbitMQConnector


class TestLLReasoner(LLReasonerBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state

    #
    # def on_observation(self, state: State, sender: str, observation: Observation, **kwargs) -> Update:
    #     state.step_counter += 1
    #
    # def on_learning_status(self, state: State, sender: str, learning_status: Any, **kwargs) -> Update:
    #     pass


class TestActuator(ActuatorBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state

    # def on_request(self, state: State, sender: str, **kwargs) -> Update:
    #     pass


class TestPerceptor(PerceptorBase):
    def step(self, state: State) -> Update:
        state.step_counter += 1

        return state

    # def on_request(self, state: State, sender: str, **kwargs) -> Update:
    #     pass


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
        connector_kwargs={
            'host': 'localhost',
            'port': 5672,
            'prefetch_count': 1
        },
        step_frequency=.5,
        exec_duration=10.,
        save_dir=Path('D:\\bsc-tmp\\phd\\phd_thesis\\hybrid-agent\\out\\test_agent_refactor'),
        save_format='json',
        resume=False,
        log_level=logging.DEBUG
    )

    asyncio.run(run_agent(test_agent))
