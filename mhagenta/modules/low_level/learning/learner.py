from typing import Any, Iterable, ClassVar

from mhagenta.utils import ModuleTypes, Outbox, ConnType, Message, Observation, State
from mhagenta.utils.common.typing import Update
from mhagenta.core.processes.mha_module import MHAModule, GlobalParams, ModuleBase


class LearnerOutbox(Outbox):
    def request_memories(self, memory_id: str = ModuleTypes.MEMORY, **kwargs) -> None:
        self._add(memory_id, ConnType.request, kwargs)

    def send_status(self, ll_reasoner_id: str, learning_status: Any, **kwargs) -> None:
        body = {'learning_status': learning_status}
        if kwargs:
            body.update(kwargs)
        self._add(ll_reasoner_id, ConnType.send, body, extension='status')

    def send_model(self, ll_reasoner_id: str, model: Any, **kwargs) -> None:
        body = {'model': model}
        if kwargs:
            body.update(kwargs)
        self._add(ll_reasoner_id, ConnType.send, body, extension='model')


class LearnerBase(ModuleBase):
    module_type: ClassVar[str] = ModuleTypes.LEARNER

    def on_task(self, state: State, sender: str, task: Any, **kwargs) -> Update:
        raise NotImplementedError()

    def on_memories(self, state: State, sender: str, observations: Iterable[Observation], **kwargs) -> Update:
        raise NotImplementedError()

    def on_model_request(self, state: State, sender: str, **kwargs) -> Update:
        raise NotImplementedError()


class Learner(MHAModule):
    def __init__(self,
                 global_params: GlobalParams,
                 base: LearnerBase
                 ):
        self._module_id = base.module_id
        self._base = base
        self._directory = global_params.directory

        out_id_channels = list()
        in_id_channels_callbacks = list()

        for ll_reasoner in self._directory.ll_reasoning:
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send, extension='model'))
            out_id_channels.append(self.sender_reg_entry(ll_reasoner, ConnType.send, extension='status'))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.request, self._receive_model_request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(ll_reasoner, ConnType.send, self._receive_task))

        for memory in self._directory.memory:
            out_id_channels.append(self.sender_reg_entry(memory, ConnType.request))
            in_id_channels_callbacks.append(self.recipient_reg_entry(memory, ConnType.send, self._receive_memories))

        super().__init__(
            global_params=global_params,
            base=base,
            out_id_channels=out_id_channels,
            in_id_channel_callbacks=in_id_channels_callbacks
        )

    def _receive_task(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received a new task {msg.id} from {sender}. Processing...')
        task = msg.body.pop('task')
        update = self._base.on_task(state=self._state, sender=sender, task=task, **msg.body)
        self.debug(f'Finished processing the new task {msg.id}!')
        return update

    def _receive_memories(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received memories {msg.id} from {sender}. Processing...')
        observations = msg.body.pop('observations')
        update = self._base.on_memories(state=self._state, sender=sender, observations=observations, **msg.body)
        self.debug(f'Finished processing memories {msg.id}!')
        return update

    def _receive_model_request(self, sender: str, channel: str, msg: Message) -> Update:
        self.info(f'Received a model request {msg.id} from {sender}. Processing...')
        update = self._base.on_model_request(state=self._state, sender=sender, **msg.body)
        self.debug(f'Finished processing the model request {msg.id}!')
        return update
