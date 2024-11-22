import logging
from typing import Any
from fastapi import FastAPI, APIRouter, Request
from httpx import Client

from mhagenta.bases import PerceptorBase, ActuatorBase
from mhagenta.states import PerceptorState, ActuatorState


class RestReceiver(PerceptorBase):
    """
    Extended receiver base class for REST-based inter-agent communication.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tags.append('restful-router')
        self.router = APIRouter()
        self.router.add_api_route('/inbox', self._on_post, methods=['POST'])

    def on_msg(self, state: PerceptorState, sender: str, msg: dict[str, Any]) -> PerceptorState:
        """
        Override to define agent's reaction to receiving a message from another agent.

        Args:
            state (PerceptorState): module's internal state enriched with relevant runtime information and
                functionality.
            sender (str): sender's `agent_id`.
            msg (dict[str, Any]): message's content.
        """
        pass

    async def _on_post(self, request: Request) -> None:
        body: dict[str, Any] = await request.json()
        sender = body.pop('sender') if 'sender' in body else 'UNKNOWN'
        self.log(logging.DEBUG, f'Received a message from {sender}!')
        self.state = self.on_msg(self.state, sender, body)


class RestSender(ActuatorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tags.append('restful-sender')
        self._client = Client()

    def __del__(self) -> None:
        self._client.close()

    def send(self, recipient_addr: Any, msg: dict[str, Any]) -> None:
        """
        Call this method to send a message to another agent.

        Args:
            recipient_addr (Any): receiver's address object. Typically can be accessed via the recipient's directory
                card (e.g. `state.directory.external[<agent_id>].address` if `agent_id` is known).
            msg (dict[str, Any]): message's content. Must be JSON serializable.
        """
        msg['sender'] = self.agent_id
        self._client.post(recipient_addr, json=msg)
