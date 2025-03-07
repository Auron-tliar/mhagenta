import dill
import asyncio
from typing import Any
import importlib.metadata
from pathlib import Path
from os import PathLike
import sys

from mhagenta.environment import MHAEnvironment


async def main(env_path: PathLike) -> None:
    with open(env_path, 'rb') as f:
        params: dict[str, Any] = dill.load(f)

    env_class: type[MHAEnvironment] = params.pop('env_class')
    env = env_class(**params)
    await env.initialize()
    await env.start()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Missing environment parameters path argument')
    path = Path(sys.argv[1]).resolve()

    print(f'Using MHAgentA version {importlib.metadata.version("mhagenta")}')
    asyncio.run(main(path))
