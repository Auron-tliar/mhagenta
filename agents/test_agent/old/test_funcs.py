from typing import Any
from pathlib import Path
import pickle
from test_template import TestTemplate
from mhagenta.defaults.connectors import RabbitMQConnector, RabbitMQAgentConnector, RabbitMQParams
from mhagenta.params import *
from mhagenta import Observation, ActionStatus


def step_func_check(state: dict[str, bool]) -> tuple[dict[str, bool], None]:
    state['step_func'] = True
    return state, None

def perc_step_func_check(state: dict[str, bool]) -> tuple[dict[str, bool], dict[str, Any]]:
    state['step_func'] = True
    return state, {}

def act_step_func_check(state: dict[str, bool]) -> tuple[dict[str, bool], dict[str, Any]]:
    state['step_func'] = True
    return state, {}

def env_call_check(state: dict[str, bool]) -> tuple[dict[str, bool], Observation]:
    state['environment_call'] = True
    return state, Observation(observation_type='test', value='test')

def act_call_check(state: dict[str, bool]) -> tuple[dict[str, bool], ActionStatus]:
    state['action_call'] = True
    return state, ActionStatus(status='test')

def process_obs_check(state: dict[str, bool], sender: str, observation: Observation) -> tuple[dict[str, bool], None]:
    state['process_observation_func'] = True
    return state, None

def process_act_check(state: dict[str, bool], sender: str, action_status: ActionStatus) -> tuple[dict[str, bool], None]:
    state['process_action_status_func'] = True
    return state, None


test_func = TestTemplate(
    agent_id='test_func_agent',
    module_connector_cls=RabbitMQConnector,
    agent_connector_cls=RabbitMQAgentConnector,
    connector_kwargs=RabbitMQParams(
        host='localhost',
        port=5672,
        prefetch_count=1),
    perceptors=PerceptorParams(
        module_id='test_perceptor',
        initial_state={
            'step_func': False,
            'environment_call': False
        },
        step_func=perc_step_func_check,
        environment_call=env_call_check,
        imports=[]),
    actuators=ActuatorParams(
        module_id='test_actuator',
        initial_state={
            'step_func': False,
            'action_call': False
        },
        step_func=act_step_func_check,
        action_call=act_call_check,
        imports=[]),
    ll_reasoner=LLParams(
        module_id='test_ll_reasoner',
        initial_state={
            'step_func': False,
            'process_observation_func': False,
            'process_action_status_func': False
        },
        step_func=step_func_check,
        process_observation_func=process_obs_check,
        process_action_status_func=process_act_check,
        process_goal_update_func=None,
        process_prediction_func=None,
        process_model_func=None,
        imports=[]),
    learner=LearnerParams(
        module_id='test_learner',
        initial_state={
            'step_func': False
        },
        step_func=step_func_check,
        process_task_func=lambda state: (state, None),
        process_memories_func=lambda state: (state, None),
        process_prediction_request_func=lambda state: (state, None),
        process_model_request_func=lambda state: (state, None),
        imports=[]),
    memory=MemoryParams(
        module_id='test_memory',
        initial_state={
            'step_func': False
        },
        step_func=step_func_check,
        process_obs_request_func=lambda state: (state, None),
        process_bel_request_func=lambda state: (state, None),
        process_observations_func=lambda state: (state, None),
        process_beliefs_func=lambda state: (state, None),
        imports=[]),
    knowledge=KnowledgeParams(
        module_id='test_knowledge',
        initial_state={
            'step_func': False
        },
        step_func=step_func_check,
        process_beliefs_func=lambda state: (state, None),
        process_belief_request_func=lambda state: (state, None),
        imports=[]),
    hl_reasoner=HLParams(
        module_id='test_hl_reasoner',
        initial_state={
            'step_func': False
        },
        step_func=step_func_check,
        process_beliefs_func=lambda state: (state, None),
        process_goals_update_func=lambda state: (state, None),
        imports=[]),
    goal_graph=GGParams(
        module_id='test_goal_graph',
        initial_state={
            'step_func': False
        },
        step_func=step_func_check,
        process_update_func=lambda state: (state, None),
        process_request_func=lambda state: (state, None),
        imports=[]),
    simulation_duration_sec=5.,
    step_frequency=1.,
    control_frequency=.5,
    status_period=4,
    start_time=-1,
    start_sync_delay= 2.,
    save_dir=r'D:\bsc-tmp\phd\phd_thesis\hybrid-agent\out',
    verbose=True
)


def check_results():
    print(f'================== Checking final states... ==================')
    print('\tPerceptor...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.perceptors.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tActuator...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.actuators.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tLL Reasoner...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.ll_reasoner.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tLearner...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.learner.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tMemory...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.memory.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tHL Reasoner...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.hl_reasoner.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tKnowledge...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.knowledge.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')

    print('\tGoal graph...')
    file = Path(test_func.save_dir) / test_func.agent_id / 'out/save' / f'{test_func.goal_graph.module_id}.state'
    with open(file, 'rb') as f:
        state = pickle.load(f)
    print(f'\t\t...{all(state.values())}')
