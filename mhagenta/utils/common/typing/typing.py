from collections.abc import Callable
from mhagenta.utils.common import State, Message


type Sender = str
type Recipient = str
type Channel = str

type StepAction = Callable[[State], State]
type MessageCallback = Callable[[Sender, Channel, Message], State]
type MsgProcessorCallback = Callable[[Sender, Channel, Message], None]
