from typing import ClassVar
from mhagenta.utils import ModuleTypes, ConnType, Message, ActionStatus, Outbox, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class ActuatorOutbox(Outbox):
    """Internal communication outbox class for Actuator.

    Used to store and process outgoing messages to other modules.

    """
    def send_status(self, ll_reasoner_id: str, status: ActionStatus, **kwargs) -> None:
        """Sends an action status report object to a low-level reasoner.

        Args:
            ll_reasoner_id (str): `module_id` of the low-level reasoner to report to.
            status (ActionStatus): action status object.
            **kwargs: additional keyword arguments to be included in the message.

        """
        body = {'action_status': status}
        if kwargs:
            body.update(kwargs)
        self._add(ll_reasoner_id, ConnType.send, body)


ActuatorState = State[ActuatorOutbox]


class ActuatorBase(ModuleBase):
    """Base class for defining Actuator behavior (also inherits common methods from `ModuleBase`).

    To implement a custom behavior, override the empty base functions: `on_init`, `step`, `on_first`, `on_last`, and/or
    reactions to messages from other modules.

    """
    module_type: ClassVar[str] = ModuleTypes.ACTUATOR

    def on_request(self, state: ActuatorState, sender: str, **kwargs) -> ActuatorState:
        """Override to define actuator's reaction to receiving an action reqeust.

        Args:
            state (ActuatorState): Actuator's internal state enriched with relevant runtime information and
                functionality.
            sender (str): `module_id` of the low-level reasoner that sent the request.
                **kwargs: additional keyword arguments included in the message.

        Returns:
            ActuatorState: modified or unaltered internal state of the module.

        """
        return state


class Actuator(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: ActuatorBase) -> None:
        self._module_id = base.module_id
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.request, self.receive_request))

        for hl_reasoner in self._directory.hl_reasoning:
            in_id_channels_callbacks.append(self.recipient_reg_entry(hl_reasoner, ConnType.request, self.receive_request))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=ActuatorOutbox
        )

    def receive_request(self, sender: str, channel: str, msg: Message) -> ActuatorState:
        self.info(f'Received action request {msg.id} from {sender}. Processing...')
        update = self._base.on_request(self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing action request {msg.id}!')
        return update
