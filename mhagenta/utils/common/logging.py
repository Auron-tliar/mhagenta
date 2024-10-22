from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging


DEFAULT_FORMAT = '[%(agent_time)f|%(mod_time)f|%(exec_time)s][%(levelname)s]::%(tags)s::%(message)s'


class LoggerExtras(BaseModel):
    agent_time: float
    mod_time: float
    exec_time: str
    tags: str


class ILogging(ABC):
    @property
    @abstractmethod
    def _logger_extras(self) -> LoggerExtras | None:
        pass

    @property
    @abstractmethod
    def _logger(self) -> logging.Logger:
        pass

    def log(self, level: int, message: str):
        self._logger.log(level, message, extra=dict() if self._logger_extras is None else self._logger_extras.model_dump())

    def debug(self, message: str):
        self._logger.debug(message, extra=dict() if self._logger_extras is None else self._logger_extras.model_dump())

    def info(self, message: str):
        self._logger.info(message, extra=dict() if self._logger_extras is None else self._logger_extras.model_dump())

    def warning(self, message: str):
        self._logger.warning(message, extra=dict() if self._logger_extras is None else self._logger_extras.model_dump())

    def error(self, message: str):
        self._logger.error(message, extra=dict() if self._logger_extras is None else self._logger_extras.model_dump())

    def critical(self, message: str):
        self._logger.critical(message, extra=dict() if self._logger_extras is None else self._logger_extras.model_dump())
