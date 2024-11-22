from typing import Any

from fastapi import APIRouter, Request


class RestEnvironmentBase:
    """
    Base class for REST-based environments.
    """
    def __init__(self, url: str) -> None:
        self.url = url
        self.tags = ['restful-router']
        self.router = APIRouter()
        self.router.add_api_route('/observation', self._on_observation_request, methods=['GET'])
        self.router.add_api_route('/action', self._on_action_request, methods=['POST'])

    def on_observation(self, **kwargs) -> dict[str, Any]:
        """
        Override to define what environment returns when observed by agents,

        Args:
            **kwargs: optional keyword parameters for observation action

        Returns:
            keyword-based observation description

        """
        return dict()

    def on_action(self, **kwargs) -> None:
        """
        Override to define the effects of an action on the environment.
        Args:
            **kwargs: keyword-based description of an action

        """
        return

    async def _on_observation_request(self, request: Request) -> dict[str, Any]:
        params = dict(request.query_params)
        params.update(request.path_params)
        return self.on_observation(**params)

    async def _on_action_request(self, request: Request) -> dict[str, Any]:
        params = await request.json()
        self.on_action(**params)
        return dict()
