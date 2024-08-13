from typing import Any
from pathlib import Path
import logging
import asyncio

from mhagenta import State, Observation, ActionStatus
from mhagenta.base import *
from mhagenta.outboxes import *
from mhagenta.utils.common import Directory
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes import MHARoot
from mhagenta.core import RabbitMQConnector


class TestLLReasoner(LLReasonerBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> LLOutbox:
        outbox = LLOutbox()
        signature = (self.module_id, func_name)
        for actuator in directory.actuation:
            outbox.request_action(actuator_id=actuator, signature=signature)
        for perceptor in directory.perception:
            outbox.request_observation(perceptor_id=perceptor, signature=signature)
        for learner in directory.learning:
            outbox.request_model(learner_id=learner, signature=signature)
            outbox.send_learner_task(learner_id=learner, task=None, signature=signature)
        for goal_graph in directory.goals:
            outbox.request_goals(goal_graph_id=goal_graph, signature=signature)
            outbox.send_goal_update(goal_graph_id=goal_graph, goals=[], signature=signature)
        for knowledge in directory.knowledge:
            outbox.send_beliefs(knowledge_id=knowledge, beliefs=[], signature=signature)
        for memory in directory.memory:
            outbox.send_memories(memory_id=memory, observations=[], signature=signature)
        return outbox


class TestActuator(ActuatorBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> ActuatorOutbox:
        outbox = ActuatorOutbox()
        signature = (self.module_id, func_name)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_status(ll_reasoner_id=ll_reasoner, status=ActionStatus(status=None), signature=signature)
        return outbox


class TestPerceptor(PerceptorBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> PerceptorOutbox:
        outbox = PerceptorOutbox()
        signature = (self.module_id, func_name)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_observation(ll_reasoner_id=ll_reasoner, observation=Observation(observation_type='none', value=None), signature=signature)
        return outbox


class TestLearner(LearnerBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> LearnerOutbox:
        outbox = LearnerOutbox()
        signature = (self.module_id, func_name)
        for memory in directory.memory:
            outbox.request_memories(memory_id=memory, signature=signature)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_status(ll_reasoner_id=ll_reasoner, status=None, signature=signature)
            outbox.send_model(ll_reasoner_id=ll_reasoner, model=None, signature=signature)
        return outbox


class TestKnowledge(KnowledgeBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> KnowledgeOutbox:
        outbox = KnowledgeOutbox()
        signature = (self.module_id, func_name)
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_beliefs(hl_reasoner_id=hl_reasoner, beliefs=[], signature=signature)
        for memory in directory.memory:
            outbox.send_memories(memory_id=memory, beliefs=[], signature=signature)
        return outbox


class TestHLReasoner(HLReasonerBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> HLOutbox:
        outbox = HLOutbox()
        signature = (self.module_id, func_name)
        for memory in directory.memory:
            outbox.request_memories(memory_id=memory, signature=signature)
        for actuator in directory.actuation:
            outbox.request_action(actuator_id=actuator, signature=signature)
        for knowledge in directory.knowledge:
            outbox.request_beliefs(knowledge_id=knowledge, signature=signature)
            outbox.send_beliefs(knowledge_id=knowledge, beliefs=[], signature=signature)
        for goal_graph in directory.goals:
            outbox.send_goals(goal_graph_id=goal_graph, goals=[], signature=signature)
        return outbox


class TestGoalGraph(GoalGraphBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> GoalGraphOutbox:
        outbox = GoalGraphOutbox()
        signature = (self.module_id, func_name)
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_goals(receiver=hl_reasoner, goals=[], signature=signature)
        for ll_reasoner in directory.ll_reasoning:
            outbox.send_goals(receiver=ll_reasoner, goals=[], signature=signature)
        return outbox


class TestMemory(MemoryBase):
    def step(self, state: State) -> Update:

        return state

    def message_all(self, directory: Directory, func_name: str) -> MemoryOutbox:
        outbox = MemoryOutbox()
        signature = (self.module_id, func_name)
        for hl_reasoner in directory.hl_reasoning:
            outbox.send_beliefs(hl_reasoner_id=hl_reasoner, beliefs=[], signature=signature)
        for learner in directory.learning:
            outbox.send_observations(learner_id=learner, observations=[], signature=signature)
        return outbox


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
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        actuators=TestActuator(
            module_id='test_actuator',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        ll_reasoners=TestLLReasoner(
            module_id='test_ll_reasoner',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        learners=TestLearner(
            module_id='test_learner',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        knowledge=TestKnowledge(
            module_id='test_knowledge',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        hl_reasoner=TestHLReasoner(
            module_id='test_hl_reasoner',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        goal_graphs=TestGoalGraph(
            module_id='test_goal_graph',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
        ),
        memory=TestMemory(
            module_id='test_memory',
            initial_state={
                'sent_from': set(),
                'received_from': set()
            }
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
