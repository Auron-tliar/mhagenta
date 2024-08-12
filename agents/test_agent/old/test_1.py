import shutil
from typing import Any
from pathlib import Path
import asyncio
import time
import numpy as np
from test_template import TestTemplate
from mhagenta.defaults.connectors import RabbitMQConnector, RabbitMQAgentConnector, RabbitMQParams
from mhagenta.params import *
from mhagenta.outboxes import *
from mhagenta import Observation, ActionStatus, Orchestrator


def ll_step_func(state: dict[str, Any]) -> tuple[dict[str, Any], LLOutbox | None]:
    state['counter'] += 1
    state['time'] = np.round(time.time(), 3)
    outbox = LLOutbox()
    if state['counter'] == 2 or state['counter'] == 10:
        outbox.request_observation(perceptor_id='test_perceptor', counter=state['counter'])
    if state['counter'] == 5 or state['counter'] == 6:
        outbox.request_action(actuator_id='test_actuator', counter=state['counter'])
    if not outbox:
        outbox = None
    return state, outbox


def ll_process_observation(state: dict[str, Any], perceptor: str, obs: Observation) -> tuple[
    dict[str, Any], LLOutbox | None]:

    state['observations'][time.time()] = f'<{perceptor}><{obs.observation_type}> {obs.value}'
    return state, None


def ll_process_action_status(state: dict[str, Any], actuator: str, status: ActionStatus) -> tuple[
    dict[str, Any], LLOutbox | None]:

    state['action_statuses'][time.time()] = f'<{actuator}> {status.status}'
    return state, None


def environment_call(state: None, counter: int) -> tuple[None, Observation]:
    return None, Observation(observation_type='sp_time', value=(-counter, time.time()))


def action_call(state: None, counter: int) -> tuple[None, ActionStatus]:
    return None, ActionStatus(status=(True, 20. * counter))


test_01 = TestTemplate(
    agent_id='test_agent_01',
    module_connector_cls=RabbitMQConnector,
    agent_connector_cls=RabbitMQAgentConnector,
    connector_kwargs=RabbitMQParams(
        host='localhost',
        port=5672,
        prefetch_count=1),
    perceptors=PerceptorParams(
        module_id='test_perceptor',
        initial_state={'counter': 0, 'time': 0.},
        step_func=None,
        environment_call=environment_call,
        imports='import time'),
    actuators=ActuatorParams(
        module_id='test_actuator',
        initial_state={'counter': 0, 'time': 0.},
        step_func=None,
        action_call=action_call,
        imports=[]),
    ll_reasoner=LLParams(
        module_id='test_ll_reasoner',
        initial_state={'counter': 0, 'time': 0., 'observations': dict(), 'action_statuses': dict()},
        step_func=ll_step_func,
        process_observation_func=ll_process_observation,
        process_action_status_func=ll_process_action_status,
        process_goal_update_func=None,
        process_prediction_func=None,
        process_model_func=None,
        imports=[
            'import time',
            'import numpy as np'
        ]),
    learner=None,
    memory=None,
    knowledge=None,
    hl_reasoner=None,
    goal_graph=None,
    simulation_duration_sec=20.,
    step_frequency=1.,
    control_frequency=.5,
    status_period=4,
    start_time=None,
    start_sync_delay= 2.,
    save_dir=r'D:\bsc-tmp\phd\phd_thesis\hybrid-agent\out',
    verbose=True
)
