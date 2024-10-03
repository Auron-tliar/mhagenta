from typing import ClassVar
from mhagenta.utils import ModuleTypes, ConnType, Message, Observation, Outbox, State
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class PerceptorOutbox(Outbox):
    def send_observation(self, ll_reasoner_id: str, observation: Observation, **kwargs) -> None:
        body = {'observation': observation}
        if kwargs:
            body.update(kwargs)
        self._add(ll_reasoner_id, ConnType.send, body)


PerceptorState = State[PerceptorOutbox]


class PerceptorBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.PERCEPTOR

    def on_request(self, state: PerceptorState, sender: str, **kwargs) -> PerceptorState:
        raise NotImplementedError()


class Perceptor(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: PerceptorBase
                 ) -> None:
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.request, self.receive_request))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks,
            outbox_cls=PerceptorOutbox
        )

    def receive_request(self, sender: str, channel: str, msg: Message) -> PerceptorState:
        self.info(f'Received observation request {msg.id} from {sender}. Processing...')
        update = self._base.on_request(self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing observation request {msg.id}!')
        return update

