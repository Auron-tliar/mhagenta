from typing import Callable, TypeAlias
from mhagenta.utils.common import State, Message


Sender: TypeAlias = str
Recipient: TypeAlias = str
Channel: TypeAlias = str

StepAction: TypeAlias = Callable[[State], State]
MessageCallback: TypeAlias = Callable[[Sender, Channel, Message], State]
MsgProcessorCallback: TypeAlias = Callable[[Sender, Channel, Message], None]
