from pydantic import BaseModel


class RabbitMQParams(BaseModel):
    host: str = 'localhost'
    port: int = 5672
    prefetch_count: int = 1
