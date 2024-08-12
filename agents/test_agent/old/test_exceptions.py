import inspect
from typing import Any, Iterable
from pathlib import Path
import pickle
from test_template import TestTemplate
from mhagenta.defaults.connectors import RabbitMQConnector, RabbitMQAgentConnector, RabbitMQParams
from mhagenta.params import *
from mhagenta import Observation, ActionStatus, ModuleTypes
from mhagenta import LLOutbox, LearnerOutbox, MemoryOutbox, KnowledgeOutbox, HLOutbox, GoalGraphOutbox, Goal, Belief


def perceptor_step_func(state: list[int]) -> tuple[list[int], dict[str, Any] | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('Perceptor test error 1')
        case 1:
            request = {'num': 1, 'sender': 'test_perceptor'}
            state[0] = 2
            return state, request
        case 2:
            state[0] = 3
            raise RuntimeError('Perceptor test error 2')
        case 3:
            request = {'num': 2, 'sender': 'test_perceptor'}
            state[0] = 4
            return state, request
        case _:
            return state, None


def actuator_step_func(state: list[int]) -> tuple[list[int], dict[str, Any] | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('Actuator test error 1')
        case 1:
            request = {'num': 1, 'sender': 'test_actuator'}
            state[0] = 2
            return state, request
        case 2:
            state[0] = 3
            raise RuntimeError('Actuator test error 2')
        case 3:
            request = {'num': 2, 'sender': 'test_actuator'}
            state[0] = 4
            return state, request
        case _:
            return state, None


def ll_message_all(num: int) -> LLOutbox:
    outbox = LLOutbox()
    outbox.request_action('test_actuator', sender=ModuleTypes.LLREASONER, num=num)
    outbox.request_observation('test_perceptor', sender=ModuleTypes.LLREASONER, num=num)
    outbox.request_prediction(num=num)
    outbox.request_model(num=num)
    outbox.request_goals(num=num)
    outbox.send_goal_update([], num=num)
    outbox.send_learner_task('task', num=num)
    outbox.send_beliefs([], num=num)
    outbox.send_memories([], num=num)
    return outbox


def learner_message_all(num: int) -> LearnerOutbox:
    outbox = LearnerOutbox()
    outbox.send_model('model', num=num)
    outbox.send_prediction('prediction', num=num)
    outbox.request_memories(num=num)
    return outbox


def memory_message_all(num: int) -> MemoryOutbox:
    outbox = MemoryOutbox()
    outbox.send_beliefs([], num=num)
    outbox.send_observations([], num=num)
    return outbox


def knowledge_message_all(num: int) -> KnowledgeOutbox:
    outbox = KnowledgeOutbox()
    outbox.send_beliefs([], num=num)
    outbox.send_memories([], num=num)
    return outbox


def hl_message_all(num: int) -> HLOutbox:
    outbox = HLOutbox()
    outbox.send_beliefs([], num=num)
    outbox.request_memories(num=num)
    outbox.request_beliefs(num=num)
    outbox.send_goals([], num=num)
    return outbox


def gg_message_all(num: int) -> GoalGraphOutbox:
    outbox = GoalGraphOutbox()
    goal = Goal(state=[], num=num)
    outbox.send_goals(ModuleTypes.HLREASONER, [goal])
    outbox.send_goals(ModuleTypes.LLREASONER, [goal])
    return outbox


def ll_step_func(state: list[int]) -> tuple[list[int], LLOutbox | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('LL test error 1')
        case 1:
            outbox = ll_message_all(1)
            state[0] = 2
            return state, outbox
        case 2:
            state[0] = 3
            raise RuntimeError('LL test error 2')
        case 3:
            outbox = ll_message_all(2)
            state[0] = 4
            return state, outbox
        case _:
            return state, None


def learner_step_func(state: list[int]) -> tuple[list[int], LearnerOutbox | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('Learner test error 1')
        case 1:
            outbox = learner_message_all(1)
            state[0] = 2
            return state, outbox
        case 2:
            state[0] = 3
            raise RuntimeError('Learner test error 2')
        case 3:
            outbox = learner_message_all(2)
            state[0] = 4
            return state, outbox
        case _:
            return state, None


def memory_step_func(state: list[int]) -> tuple[list[int], MemoryOutbox | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('Memory test error 1')
        case 1:
            outbox = memory_message_all(1)
            state[0] = 2
            return state, outbox
        case 2:
            state[0] = 3
            raise RuntimeError('Memory test error 2')
        case 3:
            outbox = memory_message_all(2)
            state[0] = 4
            return state, outbox
        case _:
            return state, None


def knowledge_step_func(state: list[int]) -> tuple[list[int], KnowledgeOutbox | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('Knowledge test error 1')
        case 1:
            outbox = knowledge_message_all(1)
            state[0] = 2
            return state, outbox
        case 2:
            state[0] = 3
            raise RuntimeError('Knowledge test error 2')
        case 3:
            outbox = knowledge_message_all(2)
            state[0] = 4
            return state, outbox
        case _:
            return state, None


def hl_step_func(state: list[int]) -> tuple[list[int], HLOutbox | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('HL test error 1')
        case 1:
            outbox = hl_message_all(1)
            state[0] = 2
            return state, outbox
        case 2:
            state[0] = 3
            raise RuntimeError('HL test error 2')
        case 3:
            outbox = hl_message_all(2)
            state[0] = 4
            return state, outbox
        case _:
            return state, None


def gg_step_func(state: list[int]) -> tuple[list[int], GoalGraphOutbox | None]:
    match state[0]:
        case 0:
            state[0] = 1
            raise RuntimeError('GG test error 1')
        case 1:
            outbox = gg_message_all(1)
            state[0] = 2
            return state, outbox
        case 2:
            state[0] = 3
            raise RuntimeError('GG test error 2')
        case 3:
            outbox = gg_message_all(2)
            state[0] = 4
            return state, outbox
        case _:
            return state, None


def environment_call(state: list[int], sender: str, num: int) -> tuple[list[int], Observation | None]:
    if sender == 'test_perceptor':
        match state[1]:
            case 0:
                state[1] = 1
                raise RuntimeError('Perceptor env call test error 1')
            case 1:
                state[1] = 2
                return state, Observation(observation_type='dict', value={num: num})
            case 2:
                state[1] = 3
                return state, Observation(observation_type='dict', value={num: num})
            case _:
                raise RuntimeError('Perceptor env call test error 3')
    else:
        raise RuntimeError('Perceptor env call test error 2')


def action_call(state: list[int], sender: str, num: int) -> tuple[list[int], ActionStatus | None]:
    if sender == 'test_actuator':
        match state[1]:
            case 0:
                state[1] = 1
                raise RuntimeError('Actuator act call test error 1')
            case 1:
                state[1] = 2
                return state, ActionStatus(status={'num': num})
            case 2:
                state[1] = 3
                return state, ActionStatus(status={'num': num})
            case _:
                raise RuntimeError('Actuator act call test error 3')
    else:
        raise RuntimeError('Actuator act call test error 2')


def ll_process_observation(state: list[int], sender: str, observation: Observation) -> tuple[list[int], LLOutbox | None]:
    state.add((sender, observation.value['func'], inspect.currentframe().f_code.co_name))
    return state, ll_message_all('ll_process_observation')


def ll_process_action_status(state: list[int], sender: str, action_status: ActionStatus) -> tuple[list[int], LLOutbox | None]:
    state.add((sender, action_status.status['func'], inspect.currentframe().f_code.co_name))
    return state, ll_message_all('ll_process_action_status')


def ll_process_goal_update(state: list[int], goals: list[Goal]) -> tuple[list[int], LLOutbox | None]:
    state.add((goals[0].misc['sender'], goals[0].misc['func'], inspect.currentframe().f_code.co_name))
    return state, ll_message_all('ll_process_goal_update')


def ll_process_prediction(state: list[int], prediction: Any, num: int) -> tuple[list[int], LLOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, ll_message_all('ll_process_prediction')


def ll_process_model(state: list[int], model: Any, num: int) -> tuple[list[int], LLOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, ll_message_all('ll_process_model')


def learner_process_task(state: list[int], task: Any, num: int) -> tuple[list[int], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, learner_message_all('learner_process_task')


def learner_process_memories(state: list[int], observations: Iterable[Observation], num: int) -> tuple[list[int], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, learner_message_all('learner_process_memories')


def learner_process_prediction_request(state: list[int], num: int) -> tuple[list[int], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, learner_message_all('learner_process_prediction_request')


def learner_process_model_request(state: list[int], num: int) -> tuple[list[int], LearnerOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, learner_message_all('learner_process_model_request')


def memory_process_obs_request(state: list[int], num: int) -> tuple[list[int], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, memory_message_all('memory_process_obs_request')


def memory_process_bel_request(state: list[int], num: int) -> tuple[list[int], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, memory_message_all('memory_process_bel_request')


def memory_process_observations(state: list[int], observations: Iterable[Observation], num: int) -> tuple[list[int], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, memory_message_all('memory_process_observations')


def memory_process_beliefs(state: list[int], beliefs: Iterable[Belief], num: int) -> tuple[list[int], MemoryOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, memory_message_all('memory_process_beliefs')


def knowledge_process_beliefs(state: list[int], sender: str, beliefs: Iterable[Belief], num: int) -> tuple[list[int], KnowledgeOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, knowledge_message_all('knowledge_process_beliefs')


def knowledge_process_belief_request(state: list[int], num: int) -> tuple[list[int], KnowledgeOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, knowledge_message_all('knowledge_process_belief_request')


def hl_process_beliefs(state: list[int], sender: str, beliefs: Iterable[Belief], num: int) -> tuple[list[int], HLOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, hl_message_all('hl_process_beliefs')


def hl_process_goal_update(state: list[int], goals: list[Goal]) -> tuple[list[int], HLOutbox | None]:
    state.add((goals[0].misc['sender'], goals[0].misc['func'], inspect.currentframe().f_code.co_name))
    return state, hl_message_all('hl_process_goal_update')


def goals_process_update(state: list[int], sender: str, goals: Iterable[Goal], num: int) -> tuple[list[int], GoalGraphOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, gg_message_all('goals_process_update')


def goals_process_request(state: list[int], num: int) -> tuple[list[int], GoalGraphOutbox | None]:
    state.add((sender, func, inspect.currentframe().f_code.co_name))
    return state, gg_message_all('goals_process_request')


test_ex = TestTemplate(
    agent_id='test_func_agent',
    module_connector_cls=RabbitMQConnector,
    agent_connector_cls=RabbitMQAgentConnector,
    connector_kwargs=RabbitMQParams(
        host='localhost',
        port=5672,
        prefetch_count=1),
    perceptors=PerceptorParams(
        module_id='test_perceptor',
        initial_state=[0, 0],
        step_func=perceptor_step_func,
        environment_call=environment_call,
        imports=['import inspect']
    ),
    actuators=ActuatorParams(
        module_id='test_actuator',
        initial_state=[0, 0],
        step_func=actuator_step_func,
        action_call=action_call,
        imports=['import inspect']
    ),
    ll_reasoner=LLParams(
        module_id='test_ll_reasoner',
        initial_state=[0, 0],
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
        initial_state=[0, 0],
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
        initial_state=[0, 0],
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
        initial_state=[0, 0],
        step_func=knowledge_step_func,
        process_beliefs_func=knowledge_process_beliefs,
        process_belief_request_func=knowledge_process_belief_request,
        extras=[knowledge_message_all],
        imports=['import inspect']
    ),
    hl_reasoner=HLParams(
        module_id='test_hl_reasoner',
        initial_state=[0, 0],
        step_func=hl_step_func,
        process_beliefs_func=hl_process_beliefs,
        process_goals_update_func=hl_process_goal_update,
        extras=[hl_message_all],
        imports=['import inspect']
    ),
    goal_graph=GGParams(
        module_id='test_goal_graph',
        initial_state=[0, 0],
        step_func=gg_step_func,
        process_update_func=goals_process_update,
        process_request_func=goals_process_request,
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
    file = Path(test_ex.save_dir) / test_ex.agent_id / 'out/save' / f'{module_id}.state'
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
        (ModuleTypes.LLREASONER, 'll_step_func', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'environment_call'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'environment_call')
    }
    successful += check_module_result(test_ex.perceptors.module_id, expected_state)

    print('\tActuator...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
        (ModuleTypes.LLREASONER, 'll_step_func', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_observation', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_action_status', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_goal_update', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_prediction', 'action_call'),
        (ModuleTypes.LLREASONER, 'll_process_model', 'action_call')
    }
    successful += check_module_result(test_ex.actuators.module_id, expected_state)

    print('\tLL Reasoner...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
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
    successful += check_module_result(test_ex.ll_reasoner.module_id, expected_state)

    print('\tLearner...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
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
    successful += check_module_result(test_ex.learner.module_id, expected_state)

    print('\tMemory...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
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
    successful += check_module_result(test_ex.memory.module_id, expected_state)

    print('\tKnowledge...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
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
    successful += check_module_result(test_ex.knowledge.module_id, expected_state)

    print('\tHL Reasoner...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
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
    successful += check_module_result(test_ex.hl_reasoner.module_id, expected_state)

    print('\tGoal graph...')
    total_tests += 1
    expected_state = {
        ('', '', ''),
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
    successful += check_module_result(test_ex.goal_graph.module_id, expected_state)

    print(f'TEST RESULTS: {successful}/{total_tests}')
