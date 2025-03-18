import sys
from typing import Any, Iterable
from pathlib import Path
import logging
from sys import argv

from mhagenta import Observation, Orchestrator
from mhagenta.bases import LLReasonerBase
from mhagenta.defaults.communication.rabbitmq.modules import RMQSenderBase, RMQReceiverBase
from mhagenta.states import *
from mhagenta.utils.common import Performatives
from mhagenta.core import RabbitMQConnector

from base import TestDataBase, check_results


MSG_TEMPLATE = 'TO({})-FROM({})-NUM({})'


class TestData(TestDataBase):
    def __init__(self, agent_id: str, external_agent_ids: Iterable[str], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._agent_id = agent_id
        self._external_agent_ids = external_agent_ids

    def expected_ll_reasoner(self, module_id: str) -> dict[str, set]:
        expected = {'wrong': set(), 'received': set(), 'sent': set()}
        for ext_id in self._external_agent_ids:
            expected['received'].update([MSG_TEMPLATE.format(self._agent_id, ext_id, i) for i in range(int(self.exec_duration // self.step_frequency))])
            expected['sent'].update([MSG_TEMPLATE.format(ext_id, self._agent_id, i) for i in range(int(self.exec_duration // self.step_frequency))])

        return expected

    def expected_perceptor(self, module_id: str) -> dict[str, set]:
        expected = {'wrong': set(), 'received': set(), 'sent': set()}
        for ext_id in self._external_agent_ids:
            expected['received'].update([MSG_TEMPLATE.format(self._agent_id, ext_id, i) for i in range(int(self.exec_duration // self.step_frequency))])

        return expected

    def expected_actuator(self, module_id: str) -> dict[str, set]:
        expected = {'wrong': set(), 'received': set(), 'sent': set()}
        for ext_id in self._external_agent_ids:
            expected['sent'].update([MSG_TEMPLATE.format(ext_id, self._agent_id, i) for i in range(int(self.exec_duration // self.step_frequency))])

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


class TestLLReasoner(LLReasonerBase):
    def step(self, state: LLState) -> LLState:
        for agent in state.directory.external:
            if agent.agent_id != self.agent_id:
                msg = MSG_TEMPLATE.format(agent.agent_id, self.agent_id, state.num_sent)
                for sender in state.directory.internal.search(['external', 'sender']):
                    state.outbox.request_action(sender.module_id, recipient=agent.agent_id, msg=msg)
                state.sent.add(msg)
        state.num_sent += 1
        return state

    def on_observation(self, state: LLState, sender: str, observation: Observation, **kwargs) -> LLState:
        if observation.content['sender'] != state.agent_id and observation.content['recipient'] == state.agent_id:
            state.received.add(observation.content['msg'])
        else:
            state.wrong.add(observation.content['msg'])
        return state


class TestRMQSender(RMQSenderBase):
    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        recipient = kwargs.pop('recipient')
        msg = kwargs.pop('msg')
        self.send(
            recipient_id=recipient,
            msg={'info': msg, 'recipient': recipient},
            performative=Performatives.INFORM
        )
        state.sent.add(msg)

        return state


class TestRMQReceiver(RMQReceiverBase):
    def on_message(self, state: PerceptorState, sender: str, msg: dict[str, Any]) -> PerceptorState:
        state.received.add(msg['info'])
        for ll_reasoner in state.directory.internal.ll_reasoning:
            state.outbox.send_observation(
                ll_reasoner.module_id,
                Observation(
                    observation_type='dict',
                    content={
                        'sender': sender,
                        'msg': msg['info'],
                        'recipient': msg['recipient']
                    }
                )
            )
        return state


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise AttributeError('Provide state save directory as a command line argument.')

    save_dir = Path(argv[1])

    agent_ids = [f'test_ext_agent_{i}' for i in range(3)]

    test_data = [TestData(save_format='dill',
                          n_perceptors=1,
                          n_actuators=1,
                          n_ll_reasoners=1,
                          n_learners=0,
                          n_knowledge=0,
                          n_hl_reasoners=0,
                          n_goal_graphs=0,
                          n_memory=0,
                          # n_rmq_senders=1,
                          # n_rmq_receivers=1,
                          agent_id=agent_id,
                          external_agent_ids=[ext_id for ext_id in agent_ids if ext_id != agent_id],
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
        agent_start_delay=100.,
        mas_rmq_close_on_exit=True
    )

    for i, agent_id in enumerate(agent_ids):
        orchestrator.add_agent(
            agent_id=agent_id,
            connector_cls=RabbitMQConnector,
            ll_reasoners=[TestLLReasoner(
                module_id=module_id,
                initial_state={
                    'sent': set(),
                    'received': set(),
                    'wrong': set(),
                    'num_sent': 0,
                    'step_counter': 20
                }
            ) for module_id in test_data[i].ll_reasoners],
            actuators=[TestRMQSender(
                agent_id=agent_id,
                module_id=module_id,
                initial_state={
                    'sent': set(),
                    'received': set(),
                    'wrong': set(),
                    'num_sent': 0,
                    'step_counter': 20
                }
            ) for module_id in test_data[i].actuators],
            perceptors=[TestRMQReceiver(
                agent_id=agent_id,
                module_id=module_id,
                initial_state={
                    'sent': set(),
                    'received': set(),
                    'wrong': set(),
                    'num_sent': 0,
                    'step_counter': 20
                }
            ) for module_id in test_data[i].perceptors]
        )

    orchestrator.run(force_run=True, local_build=Path('C:\\phd\\mhagenta\\mhagenta').resolve())

    for i, agent_id in enumerate(agent_ids):
        check_results(
            agent_id=agent_id,
            save_dir=orchestrator[agent_id].save_dir,
            test_data=test_data[i],
            fields=['sent', 'received', 'wrong'],
            extra_ok=True
        )
