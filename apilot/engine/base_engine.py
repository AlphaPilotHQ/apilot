"""
Base Engine Module

Defines the abstract BaseEngine class for function engines.
"""

from abc import ABC, abstractmethod

from apilot.core.event import EventEngine

ENGINE_REGISTRY = []


def register_engine(cls):
    ENGINE_REGISTRY.append(cls)
    return cls


class BaseEngine(ABC):
    """
    Abstract class for implementing a function engine.
    """

    def __init__(
        self,
        main_engine,
        event_engine: EventEngine,
        engine_name: str,
    ) -> None:
        self.main_engine = main_engine
        self.event_engine = event_engine
        self.engine_name = engine_name

    @abstractmethod
    def close(self):
        """
        Close the engine and release resources.
        """
        pass
