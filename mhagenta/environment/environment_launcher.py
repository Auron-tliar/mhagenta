import os
import dill
import asyncio
from typing import Any
import importlib.metadata
from pathlib import Path
from os import PathLike
import sys
import signal
import functools

from mhagenta.environment import MHAEnvironment


async def main() -> None:
    with open(Path('/agent/env_params').as_posix(), 'rb') as f:
        params: dict[str, Any] = dill.load(f)

    id_override = os.environ.get('AGENT_ID')
    if id_override is not None and id_override != '':
        params['env_id'] = os.environ.get('AGENT_ID')

    env_class: type[MHAEnvironment] = params.pop('env_class')
    env = env_class(**params)

    await env.initialize()
    await env.start()


if __name__ == '__main__':
    asyncio.run(main())
