from typing import Any, Callable, TypeAlias, ParamSpec
from mhagenta.utils.common import State, Message, Outbox


P = ParamSpec('P')
Sender: TypeAlias = str
Recipient: TypeAlias = str
Channel: TypeAlias = str

# Update: TypeAlias = tuple[State, Outbox | None] | State
StepAction: TypeAlias = Callable[[State], State]
# MsgCallbackBase: TypeAlias = Callable[[Sender, Channel, Message], Any]
MessageCallback: TypeAlias = Callable[[Sender, Channel, Message], State]
MsgProcessorCallback: TypeAlias = Callable[[Sender, Channel, Message], None]
