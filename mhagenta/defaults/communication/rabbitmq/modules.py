from typing import Any

from mhagenta.bases import ActuatorBase, PerceptorBase
from mhagenta.core import RabbitMQConnector
from mhagenta.utils import Message





class RMQSender(ActuatorBase):
    """
    Extended receiver (Perceptor) base class for REST-based inter-agent communication.
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







class RMQReceiverBase(PerceptorBase):
    """
    Extended sender (Actuator) base class for inter-agent communication.
    """
    def __init__(self, host: str = 'localhost', port: int = 5672, exchange_name: str = 'mhagenta', **kwargs):
        super().__init__(**kwargs)
        self.tags.append('external-sender')
        self._ext_messenger = RabbitMQConnector(
            agent_id=self.agent_id,
            sender_id=self._agent_id,
            agent_time=self.state.agent_time,
            host=host,
            port=port,
            log_tags=[self.agent_id, 'ExternalReceiver'],
            external_exchange_name=exchange_name,
        )

    def __del__(self) -> None:
        self._ext_messenger.stop()

    def send(self, recipient_id: str, msg: dict[str, Any]) -> None:
        """
        Call this method to send a message to another agent.

        Args:
            recipient_id (Any): receiver's address object. Typically, can be accessed via the recipient's directory
                card (e.g. `state.directory.external[<agent_id>].address` if `agent_id` is known).
            msg (dict[str, Any]): message's content. Must be JSON serializable.
        """
        msg['sender'] = self.agent_id
        self._ext_messenger.send(
            recipient=recipient_id,
            channel=recipient_id,
            msg=Message(
                body=msg,
                sender_id=self.agent_id,
                recipient_id=recipient_id,
                ts=self.state.time,
                performative='info'
            )
        )
