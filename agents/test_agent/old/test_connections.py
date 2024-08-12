import inspect
from pprint import pprint
from typing import Any, Iterable
from pathlib import Path
import pickle
from test_template import TestTemplate
from mhagenta.defaults.connectors import RabbitMQConnector, RabbitMQAgentConnector, RabbitMQParams
from mhagenta.params import *
from mhagenta import Observation, ActionStatus, ModuleTypes
from mhagenta import LLOutbox, LearnerOutbox, MemoryOutbox, KnowledgeOutbox, HLOutbox, GoalGraphOutbox, Goal, Belief


def perceptor_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], dict[str, Any] | None]:
    if ('', '', '') not in state:
        request = {
            'sender': 'test_perceptor',
            'func': 'perceptor_step_func'
        }
        state.add(('', '', ''))
        return state, request
    else:
        return state, None


def actuator_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], dict[str, Any] | None]:
    if ('', '', '') not in state:
        request = {
            'sender': 'test_actuator',
            'func': 'actuator_step_func'
        }
        state.add(('', '', ''))

        return state, request
    else:
        return state, None


def ll_message_all(func: str) -> LLOutbox:
    outbox = LLOutbox()
    outbox.request_action('test_actuator', sender=ModuleTypes.LLREASONER, func=func)
    outbox.request_observation('test_perceptor', sender=ModuleTypes.LLREASONER, func=func)
    outbox.request_prediction(sender=ModuleTypes.LLREASONER, func=func)
    outbox.request_model(sender=ModuleTypes.LLREASONER, func=func)
    outbox.request_goals(func=func)
    outbox.send_goal_update([], func=func)
    outbox.send_learner_task('task', sender=ModuleTypes.LLREASONER, func=func)
    outbox.send_beliefs([], func=func)
    outbox.send_memories([], sender=ModuleTypes.LLREASONER, func=func)
    return outbox


def learner_message_all(func: str) -> LearnerOutbox:
    outbox = LearnerOutbox()
    outbox.send_model('model', sender=ModuleTypes.LEARNER, func=func)
    outbox.send_prediction('prediction', sender=ModuleTypes.LEARNER, func=func)
    outbox.request_memories(sender=ModuleTypes.LEARNER, func=func)
    return outbox


def memory_message_all(func: str) -> MemoryOutbox:
    outbox = MemoryOutbox()
    outbox.send_beliefs([], func=func)
    outbox.send_observations([], sender=ModuleTypes.MEMORY, func=func)
    return outbox


def knowledge_message_all(func: str) -> KnowledgeOutbox:
    outbox = KnowledgeOutbox()
    outbox.send_beliefs([], func=func)
    outbox.send_memories([], sender=ModuleTypes.KNOWLEDGE, func=func)
    return outbox


def hl_message_all(func: str) -> HLOutbox:
    outbox = HLOutbox()
    outbox.send_beliefs([], func=func)
    outbox.request_memories(sender=ModuleTypes.HLREASONER, func=func)
    outbox.request_beliefs(sender=ModuleTypes.HLREASONER, func=func)
    outbox.send_goals([], func=func)
    return outbox


def gg_message_all(func: str) -> GoalGraphOutbox:
    outbox = GoalGraphOutbox()
    goal = Goal(state=[], sender=ModuleTypes.GOALGRAPH, func=func)
    outbox.send_goals(ModuleTypes.HLREASONER, [goal])
    outbox.send_goals(ModuleTypes.LLREASONER, [goal])
    return outbox


def ll_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], LLOutbox | None]:
    if ('', '', '') not in state:
        outbox = ll_message_all('ll_step_func')
        state.add(('', '', ''))
        return state, outbox
    else:
        return state, None


def learner_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], LearnerOutbox | None]:
    if ('', '', '') not in state:
        outbox = learner_message_all('learner_step_func')
        state.add(('', '', ''))
        return state, outbox
    else:
        return state, None


def memory_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], MemoryOutbox | None]:
    if ('', '', '') not in state:
        outbox = memory_message_all('memory_step_func')
        state.add(('', '', ''))
        return state, outbox
    else:
        return state, None


def knowledge_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], KnowledgeOutbox | None]:
    if ('', '', '') not in state:
        outbox = knowledge_message_all('knowledge_step_func')
        state.add(('', '', ''))
        return state, outbox
    else:
        return state, None


def hl_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], HLOutbox | None]:
    if ('', '', '') not in state:
        outbox = hl_message_all('hl_step_func')
        state.add(('', '', ''))
        return state, outbox
    else:
        return state, None


def gg_step_func(state: set[tuple[str, str, str]]) -> tuple[set[tuple[str, str, str]], GoalGraphOutbox | None]:
    if ('', '', '') not in state:
        outbox = gg_message_all('goal_graph_step_func')
        state.add(('', '', ''))
        return state, outbox
    else:
        return state, None


def environment_call(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], Observation | None]:
    if sender == 'test_perceptor' and ('_', '_', '_') not in state:
        state.add(('_', '_', '_'))
        return state, Observation(observation_type='dict', value={'sender': sender, 'func': func})
    else:
        state.add((sender, func, 'environment_call'))
        return state, Observation(observation_type='dict', value={'sender': 'test_perceptor', 'func': 'environment_call'})


def action_call(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], ActionStatus | None]:
    if sender == 'test_actuator' and ('_', '_', '_') not in state:
        state.add(('_', '_', '_'))
        return state, ActionStatus(status={'sender': sender, 'func': func})
    else:
        state.add((sender, func, 'action_call'))
        return state, ActionStatus(status={'sender': 'test_actuator', 'func': 'action_call'})


def ll_process_observation(state: set[tuple[str, str, str]], sender: str, observation: Observation) -> tuple[set[tuple[str, str, str]], LLOutbox | None]:
    state.add((sender, observation.value['func'], inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = ll_message_all('ll_process_observation')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def ll_process_action_status(state: set[tuple[str, str, str]], sender: str, action_status: ActionStatus) -> tuple[set[tuple[str, str, str]], LLOutbox | None]:
    state.add((sender, action_status.status['func'], inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = ll_message_all('ll_process_action_status')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def ll_process_goal_update(state: set[tuple[str, str, str]], goals: list[Goal]) -> tuple[set[tuple[str, str, str]], LLOutbox | None]:
    state.add((goals[0].misc['sender'], goals[0].misc['func'], inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = ll_message_all('ll_process_goal_update')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def ll_process_prediction(state: set[tuple[str, str, str]], prediction: Any, sender: str, func: str) -> tuple[set[tuple[str, str, str]], LLOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = ll_message_all('ll_process_prediction')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def ll_process_model(state: set[tuple[str, str, str]], model: Any, sender: str, func: str) -> tuple[set[tuple[str, str, str]], LLOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = ll_message_all('ll_process_model')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def learner_process_task(state: set[tuple[str, str, str]], task: Any, sender: str, func: str) -> tuple[set[tuple[str, str, str]], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = learner_message_all('learner_process_task')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def learner_process_memories(state: set[tuple[str, str, str]], observations: Iterable[Observation], sender: str, func: str) -> tuple[set[tuple[str, str, str]], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = learner_message_all('learner_process_memories')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def learner_process_prediction_request(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = learner_message_all('learner_process_prediction_request')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def learner_process_model_request(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = learner_message_all('learner_process_model_request')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def memory_process_obs_request(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = memory_message_all('memory_process_obs_request')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def memory_process_bel_request(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = memory_message_all('memory_process_bel_request')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def memory_process_observations(state: set[tuple[str, str, str]], observations: Iterable[Observation], sender: str, func: str) -> tuple[set[tuple[str, str, str]], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = memory_message_all('memory_process_observations')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def memory_process_beliefs(state: set[tuple[str, str, str]], beliefs: Iterable[Belief], sender: str, func: str) -> tuple[set[tuple[str, str, str]], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = memory_message_all('memory_process_beliefs')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def knowledge_process_beliefs(state: set[tuple[str, str, str]], sender: str, beliefs: Iterable[Belief], func: str) -> tuple[set[tuple[str, str, str]], KnowledgeOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = knowledge_message_all('knowledge_process_beliefs')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def knowledge_process_belief_request(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], KnowledgeOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = knowledge_message_all('knowledge_process_belief_request')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def hl_process_beliefs(state: set[tuple[str, str, str]], sender: str, beliefs: Iterable[Belief], func: str) -> tuple[set[tuple[str, str, str]], HLOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = hl_message_all('hl_process_beliefs')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def hl_process_goal_update(state: set[tuple[str, str, str]], goals: list[Goal]) -> tuple[set[tuple[str, str, str]], HLOutbox | None]:
    state.add((goals[0].misc['sender'], goals[0].misc['func'], inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = hl_message_all('hl_process_goal_update')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def gg_process_update(state: set[tuple[str, str, str]], sender: str, goals: Iterable[Goal], func: str) -> tuple[set[tuple[str, str, str]], GoalGraphOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = gg_message_all('goals_process_update')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


def gg_process_request(state: set[tuple[str, str, str]], sender: str, func: str) -> tuple[set[tuple[str, str, str]], GoalGraphOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    if ('_', '_', '_') not in state:
        outbox = gg_message_all('goals_process_request')
        state.add(('_', '_', '_'))
    else:
        outbox = None
    return state, outbox


test_conn = TestTemplate(
    agent_id='test_func_agent',
    module_connector_cls=RabbitMQConnector,
    agent_connector_cls=RabbitMQAgentConnector,
    connector_kwargs=RabbitMQParams(
        host='localhost',
        port=5672,
        prefetch_count=1),
    perceptors=PerceptorParams(
        module_id='test_perceptor',
        initial_state=set(),
        step_func=perceptor_step_func,
        environment_call=environment_call,
        imports=['import inspect']
    ),
    actuators=ActuatorParams(
        module_id='test_actuator',
        initial_state=set(),
        step_func=actuator_step_func,
        action_call=action_call,
        imports=['import inspect']
    ),
    ll_reasoner=LLParams(
        module_id='test_ll_reasoner',
        initial_state=set(),
        step_func=ll_step_func,
        process_observation_func=ll_process_observation,
        process_action_status_func=ll_process_action_status,
        process_goal_update_func=ll_process_goal_update,
        process_prediction_func=ll_process_prediction,
        process_model_func=ll_process_model,
        extras=[ll_message_all],
        imports=['import inspect']
    ),
    learner=LearnerParams(
        module_id='test_learner',
        initial_state=set(),
        step_func=learner_step_func,
        process_task_func=learner_process_task,
        process_memories_func=learner_process_memories,
        process_prediction_request_func=learner_process_prediction_request,
        process_model_request_func=learner_process_model_request,
        extras=[learner_message_all],
        imports=['import inspect']
    ),
    memory=MemoryParams(
        module_id='test_memory',
        initial_state=set(),
        step_func=memory_step_func,
        process_obs_request_func=memory_process_obs_request,
        process_bel_request_func=memory_process_bel_request,
        process_observations_func=memory_process_observations,
        process_beliefs_func=memory_process_beliefs,
        extras=[memory_message_all],
        imports=['import inspect']
    ),
    knowledge=KnowledgeParams(
        module_id='test_knowledge',
        initial_state=set(),
        step_func=knowledge_step_func,
        process_beliefs_func=knowledge_process_beliefs,
        process_belief_request_func=knowledge_process_belief_request,
        extras=[knowledge_message_all],
        imports=['import inspect']
    ),
    hl_reasoner=HLParams(
        module_id='test_hl_reasoner',
        initial_state=set(),
        step_func=hl_step_func,
        process_beliefs_func=hl_process_beliefs,
        process_goals_update_func=hl_process_goal_update,
        extras=[hl_message_all],
        imports=['import inspect']
    ),
    goal_graph=GGParams(
        module_id='test_goal_graph',
        initial_state=set(),
        step_func=gg_step_func,
        process_update_func=gg_process_update,
        process_request_func=gg_process_request,
        extras=[gg_message_all],
        imports=['import inspect']
    ),
    simulation_duration_sec=5.,
    step_frequency=1.,
    control_frequency=.5,
    status_period=4,
    start_time=-1,
    start_sync_delay= 2.,
    save_dir=r'D:\bsc-tmp\phd\phd_thesis\hybrid-agent\out',
    verbose=True
)


def check_module_result(module_id: str, expected_state: set[tuple[str, str, str]]) -> int:
    print(f'\t{module_id}...')
    file = Path(test_conn.save_dir) / test_conn.agent_id / 'out/save' / f'{module_id}.state'
    with open(file, 'rb') as f:
        state: set = pickle.load(f)
        result = (state == expected_state)
        if result:
            print('\t\t...SUCCEEDED!')
            return 1
        else:
            print('\t\t...FAILED!')
            print('\t\t\tUnexpected connection records:')
            print(f'\t\t\t{"\n\t\t\t".join([str(item) for item in state.difference(expected_state)])}')
            print('\t\t\tMissing connection records:')
            print(f'\t\t\t{"\n\t\t\t".join([str(item) for item in expected_state.difference(state)])}')
            print()
            return 0


def check_results():
    print(f'================== Checking final states... ==================')
    total_tests = 0
    successful = 0

    print('\tPerceptor...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'environment_call')
    }
    successful += check_module_result(test_conn.perceptors.module_id, expected_state)

    print('\tActuator...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'action_call')
    }
    successful += check_module_result(test_conn.actuators.module_id, expected_state)

    print('\tLL Reasoner...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        ('test_actuator', 'actuator_step_func', 'll_process_action_status'),
        ('test_actuator', 'action_call', 'll_process_action_status'),
        ('test_perceptor', 'perceptor_step_func', 'll_process_observation'),
        ('test_perceptor', 'environment_call', 'll_process_observation'),
        (ModuleTypes.LEARNER, 'learner_step_func', 'll_process_model'),
        (ModuleTypes.LEARNER, 'learner_step_func', 'll_process_prediction'),
        (ModuleTypes.LEARNER, 'learner_process_task', 'll_process_model'),
        (ModuleTypes.LEARNER, 'learner_process_task', 'll_process_prediction'),
        (ModuleTypes.LEARNER, 'learner_process_prediction_request', 'll_process_model'),
        (ModuleTypes.LEARNER, 'learner_process_prediction_request', 'll_process_prediction'),
        (ModuleTypes.LEARNER, 'learner_process_model_request', 'll_process_model'),
        (ModuleTypes.LEARNER, 'learner_process_model_request', 'll_process_prediction'),
        (ModuleTypes.LEARNER, 'learner_process_memories', 'll_process_model'),
        (ModuleTypes.LEARNER, 'learner_process_memories', 'll_process_prediction'),
        (ModuleTypes.GOALGRAPH, 'goal_graph_step_func', 'll_process_goal_update'),
        (ModuleTypes.GOALGRAPH, 'goals_process_update', 'll_process_goal_update'),
        (ModuleTypes.GOALGRAPH, 'goals_process_request', 'll_process_goal_update')
    }
    successful += check_module_result(test_conn.ll_reasoner.module_id, expected_state)

    print('\tLearner...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'learner_process_task'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'learner_process_prediction_request'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'learner_process_model_request'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'learner_process_task'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'learner_process_prediction_request'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'learner_process_model_request'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'learner_process_task'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'learner_process_prediction_request'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'learner_process_model_request'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'learner_process_task'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'learner_process_prediction_request'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'learner_process_model_request'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'learner_process_task'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'learner_process_prediction_request'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'learner_process_model_request'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'learner_process_task'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'learner_process_prediction_request'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'learner_process_model_request'),
        (ModuleTypes.MEMORY, 'memory_step_func', 'learner_process_memories'),
        (ModuleTypes.MEMORY, 'memory_process_observations', 'learner_process_memories'),
        (ModuleTypes.MEMORY, 'memory_process_beliefs', 'learner_process_memories'),
        (ModuleTypes.MEMORY, 'memory_process_obs_request', 'learner_process_memories'),
        (ModuleTypes.MEMORY, 'memory_process_bel_request', 'learner_process_memories')
    }
    successful += check_module_result(test_conn.learner.module_id, expected_state)

    print('\tMemory...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'memory_process_observations'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'memory_process_observations'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'memory_process_observations'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'memory_process_observations'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'memory_process_observations'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'memory_process_observations'),
        (ModuleTypes.KNOWLEDGE, 'knowledge_step_func', 'memory_process_beliefs'),
        (ModuleTypes.KNOWLEDGE, 'knowledge_process_beliefs', 'memory_process_beliefs'),
        (ModuleTypes.KNOWLEDGE, 'knowledge_process_belief_request', 'memory_process_beliefs'),
        (ModuleTypes.LEARNER, 'learner_step_func', 'memory_process_obs_request'),
        (ModuleTypes.LEARNER, 'learner_process_task', 'memory_process_obs_request'),
        (ModuleTypes.LEARNER, 'learner_process_prediction_request', 'memory_process_obs_request'),
        (ModuleTypes.LEARNER, 'learner_process_model_request', 'memory_process_obs_request'),
        (ModuleTypes.LEARNER, 'learner_process_memories', 'memory_process_obs_request'),
        (ModuleTypes.HLREASONER, 'hl_step_func', 'memory_process_bel_request'),
        (ModuleTypes.HLREASONER, 'hl_process_beliefs', 'memory_process_bel_request'),
        (ModuleTypes.HLREASONER, 'hl_process_goal_update', 'memory_process_bel_request')
    }
    successful += check_module_result(test_conn.memory.module_id, expected_state)

    print('\tKnowledge...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'knowledge_process_beliefs'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'knowledge_process_beliefs'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'knowledge_process_beliefs'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'knowledge_process_beliefs'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'knowledge_process_beliefs'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'knowledge_process_beliefs'),
        (ModuleTypes.HLREASONER, 'hl_step_func', 'knowledge_process_beliefs'),
        (ModuleTypes.HLREASONER, 'hl_step_func', 'knowledge_process_belief_request'),
        (ModuleTypes.HLREASONER, 'hl_process_beliefs', 'knowledge_process_beliefs'),
        (ModuleTypes.HLREASONER, 'hl_process_beliefs', 'knowledge_process_belief_request'),
        (ModuleTypes.HLREASONER, 'hl_process_goal_update', 'knowledge_process_beliefs'),
        (ModuleTypes.HLREASONER, 'hl_process_goal_update', 'knowledge_process_belief_request')
    }
    successful += check_module_result(test_conn.knowledge.module_id, expected_state)

    print('\tHL Reasoner...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.KNOWLEDGE, 'knowledge_step_func', 'hl_process_beliefs'),
        (ModuleTypes.KNOWLEDGE, 'knowledge_process_beliefs', 'hl_process_beliefs'),
        (ModuleTypes.KNOWLEDGE, 'knowledge_process_belief_request', 'hl_process_beliefs'),
        (ModuleTypes.MEMORY, 'memory_step_func', 'hl_process_beliefs'),
        (ModuleTypes.MEMORY, 'memory_process_observations', 'hl_process_beliefs'),
        (ModuleTypes.MEMORY, 'memory_process_beliefs', 'hl_process_beliefs'),
        (ModuleTypes.MEMORY, 'memory_process_obs_request', 'hl_process_beliefs'),
        (ModuleTypes.MEMORY, 'memory_process_bel_request', 'hl_process_beliefs'),
        (ModuleTypes.GOALGRAPH, 'goal_graph_step_func', 'hl_process_goal_update'),
        (ModuleTypes.GOALGRAPH, 'goals_process_update', 'hl_process_goal_update'),
        (ModuleTypes.GOALGRAPH, 'goals_process_request', 'hl_process_goal_update')
    }
    successful += check_module_result(test_conn.hl_reasoner.module_id, expected_state)

    print('\tGoal graph...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        ('_', '_', '_'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'goals_process_update'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'goals_process_update'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'goals_process_update'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'goals_process_update'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'goals_process_update'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'goals_process_update'),
        (ModuleTypes.HLREASONER, 'hl_step_func', 'goals_process_update'),
        (ModuleTypes.HLREASONER, 'hl_process_beliefs', 'goals_process_update'),
        (ModuleTypes.HLREASONER, 'hl_process_goal_update', 'goals_process_update'),
        (ModuleTypes.LLREASONER, 'll_step_func', 'goals_process_request'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'goals_process_request'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'goals_process_request'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'goals_process_request'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'goals_process_request'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'goals_process_request')
    }
    successful += check_module_result(test_conn.goal_graph.module_id, expected_state)

    print(f'TEST RESULTS: {successful}/{total_tests}')
