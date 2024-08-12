import asyncio
import os
from pathlib import Path
import shutil
import logging
from mhagenta import Orchestrator

from agents.test_agent.test_template import TestTemplate
from agents.test_agent.test_1 import test_01 as test_params
# from agents.test_agent.test_funcs import test_func as test_params
# from agents.test_agent.test_connections import test_conn as test_params
# from agents.test_agent.test_funcs import check_results
# from agents.test_agent.test_connections import check_results


async def run_test(test: TestTemplate, save_dir: str | os.PathLike = r'D:\bsc-tmp\phd\phd_thesis\hybrid-agent\out'):
    save_dir = Path(save_dir).absolute()
    for sub in save_dir.iterdir():
        shutil.rmtree(sub)

    orchestrator = Orchestrator(
        save_dir=test.save_dir,
        log_level=logging.DEBUG,
        simulation_duration_sec=test.simulation_duration_sec,
        step_frequency=test.step_frequency,
        control_frequency=test.control_frequency,
        status_period=test.status_period,
        start_time=test.start_time,
        start_sync_delay=test.start_sync_delay
    )

    orchestrator.add_agent(
        agent_id=test.agent_id,
        module_connector_cls=test.module_connector_cls,
        agent_connector_cls=test.agent_connector_cls,
        connector_kwargs=test.connector_kwargs,
        perceptors=test.perceptors,
        actuators=test.actuators,
        ll_reasoner=test.ll_reasoner,
        learner=test.learner,
        memory=test.memory,
        knowledge=test.knowledge,
        hl_reasoner=test.hl_reasoner,
        goal_graph=test.goal_graph
    )

    await orchestrator.start(force_run=True)


if __name__ == "__main__":
    asyncio.run(run_test(test_params))
    # check_results()

