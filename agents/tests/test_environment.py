import sys
from typing import Any, Iterable
from pathlib import Path
import logging
from sys import argv

from mhagenta import Observation, Orchestrator, ActionStatus
from mhagenta.bases import LLReasonerBase
from mhagenta.defaults.communication.rabbitmq.modules import RMQPerceptorBase, RMQActuatorBase
from mhagenta.environment import MHAEnvBase
from mhagenta.states import *
from mhagenta.utils.common import Performatives
from mhagenta.core import RabbitMQConnector

from base import TestDataBase, check_results


REQUEST_TEMPLATE = 'OBS+REQUEST-AT-STEP-[{}]-BY[{}]'
OBSERVATION_TEMPLATE = 'OBSERVATION[{}]-FOR-AGENT[{}]'
ACTION_TEMPLATE = 'ACTION[{}]-BY-AGENT[{}]'


class TestData(TestDataBase):
    def __init__(self, agent_id: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._agent_id = agent_id

    def expected_ll_reasoner(self, module_id: str) -> dict[str, set]:
        expected = {
            'observations_requested': set([REQUEST_TEMPLATE.format(i, self._agent_id) for i in range(int(self.exec_duration // self.step_frequency))]),
            'observations_received': set([OBSERVATION_TEMPLATE.format(i, self._agent_id) for i in range(int(self.exec_duration // self.step_frequency))]),
            'actions_sent': set([ACTION_TEMPLATE.format(i, self._agent_id) for i in range(int(self.exec_duration // self.step_frequency))])
        }

        return expected

    def expected_perceptor(self, module_id: str) -> dict[str, set]:
        expected = {
            'observations_requested': set([REQUEST_TEMPLATE.format(i, self._agent_id) for i in range(int(self.exec_duration // self.step_frequency))]),
            'observations_received': set([OBSERVATION_TEMPLATE.format(i, self._agent_id) for i in range(int(self.exec_duration // self.step_frequency))]),
            'actions_sent': set()
        }

        return expected

    def expected_actuator(self, module_id: str) -> dict[str, set]:
        expected = {
            'observations_requested': set(),
            'observations_received': set(),
            'actions_sent': set([ACTION_TEMPLATE.format(i, self._agent_id) for i in range(int(self.exec_duration // self.step_frequency))])
        }

        return expected

    def expected_learner(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_knowledge(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_hl_reasoner(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_goal_graph(self, module_id: str) -> dict[str, set]:
        return {}

    def expected_memory(self, module_id: str) -> dict[str, set]:
        return {}


class TestMHAEnvironment(MHAEnvBase):
    def on_observe(self, state: dict[str, Any], sender_id: str, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        state['observation_requests'].add(kwargs['request'])
        return state, {'observation': OBSERVATION_TEMPLATE.format(state['values'][sender_id], sender_id)}

    def on_action(self, state: dict[str, Any], sender_id: str, **kwargs) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any] | None]:
        state['action_requests'].add(kwargs['request'])
        # ind = int(sender_id.split('_')[-1])
        state['values'][sender_id] += 1
        if all([val >= state['step'] for val in state['values'].values()]):
            state['step'] += 1
        return state



class TestLLReasoner(LLReasonerBase):
    def step(self, state: LLState) -> LLState:
        request = REQUEST_TEMPLATE.format(state.step_counter, state.agent_id)
        state.outbox.request_observation(state.directory.internal.search(['env-perceptor'])[0].module_id, request=request)
        state.observations_requested.add(request)
        return state

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs) -> LLState:
        state.observations_received.add(observation.content)
        request = ACTION_TEMPLATE.format(state.step_counter, state.agent_id)
        state.outbox.request_action(state.directory.internal.search(['env-actuator'])[0].module_id, request=request)
        return state

    def on_action_status(self, state: LLState, sender: str, action_status: ActionStatus, **kwargs) -> LLState:
        state.actions_sent.add(action_status.status)
        state.step_counter += 1
        return state


class TestRMQPerceptor(RMQPerceptorBase):
    def on_request(self, state: PerceptorState, sender: str, **kwargs) -> PerceptorState:
        state.observations_requested.add(kwargs['request'])
        self.observe(request=kwargs['request'])

        return state

    def on_observation(self, state: PerceptorState, **kwargs) -> PerceptorState:
        state.observations_received.add(kwargs['observation'])
        state.outbox.send_observation(state.directory.internal.ll_reasoning[0].module_id, Observation(observation_type='str', content=kwargs['observation']))

        state.step_counter += 1

        return state


class TestRMQActuator(RMQActuatorBase):
    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        state.actions_sent.add(kwargs['request'])
        self.act(request=kwargs['request'])

        state.outbox.send_status(state.directory.internal.ll_reasoning[0].module_id, ActionStatus(status=kwargs['request']))
        state.step_counter += 1

        return state


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Provide state save directory as a command line argument.')

    save_dir = Path(argv[1])

    agent_ids = [f'test_env_agent_{i}' for i in range(1)]

    test_data = [TestData(save_format='dill',
                          n_perceptors=1,
                          n_actuators=1,
                          n_ll_reasoners=1,
                          n_learners=0,
                          n_knowledge=0,
                          n_hl_reasoners=0,
                          n_goal_graphs=0,
                          n_memory=0,
                          agent_id=agent_id
                          ) for i, agent_id in enumerate(agent_ids)]

    orchestrator = Orchestrator(
        save_dir=save_dir,
        connector_cls=RabbitMQConnector,
        connector_kwargs={
            'host': 'localhost',
            'port': 5672,
            'prefetch_count': 1
        },
        step_frequency=test_data[0].step_frequency,
        exec_duration=test_data[0].exec_duration + 0.1,
        save_format=test_data[0].save_format,
        resume=False,
        log_level=logging.DEBUG,
        mas_rmq_uri='localhost:5672',
        agent_start_delay=70.,
        mas_rmq_close_on_exit=True,
        mas_rmq_exchange_name='mhagenta-env'
    )

    orchestrator.set_environment(
        base=TestMHAEnvironment(init_state={
            'values': {
                agent_id: 0 for agent_id in agent_ids
            },
            'step': 0,
            'action_requests': set(),
            'observation_requests': set()
        })
    )

    for i, agent_id in enumerate(agent_ids):
        orchestrator.add_agent(
            agent_id=agent_id,
            connector_cls=RabbitMQConnector,
            ll_reasoners=[TestLLReasoner(
                module_id=module_id,
                initial_state={
                    'observations_requested': set(),
                    'observations_received': set(),
                    'actions_sent': set(),
                    'step_counter': 0
                }
            ) for module_id in test_data[i].ll_reasoners],
            actuators=[TestRMQActuator(
                agent_id=agent_id,
                module_id=module_id,
                initial_state={
                    'observations_requested': set(),
                    'observations_received': set(),
                    'actions_sent': set(),
                    'step_counter': 0
                }
            ) for module_id in test_data[i].actuators],
            perceptors=[TestRMQPerceptor(
                agent_id=agent_id,
                module_id=module_id,
                initial_state={
                    'observations_requested': set(),
                    'observations_received': set(),
                    'actions_sent': set(),
                    'step_counter': 0
                }
            ) for module_id in test_data[i].perceptors]
        )

    orchestrator.run(
        force_run=True,
        local_build=Path('C:\\phd\\mhagenta\\mhagenta').resolve(),
        rebuild_agents=True
    )

    for i, agent_id in enumerate(agent_ids):
        check_results(
            agent_id=agent_id,
            save_dir=orchestrator[agent_id].save_dir,
            test_data=test_data[i],
            fields=['observations_requested', 'observations_received', 'actions_sent'],
            extra_ok=True
        )
