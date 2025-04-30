from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from time import sleep
from typing import Any, TypeAlias

from .constant import (
    EVENT_TIMER,
)

import logging
logger = logging.getLogger("EventEngine")

@dataclass
class Event:
    type: str
    data: Any = None

# Type alias for event handler functions
HandlerType: TypeAlias = Callable[[Event], None]


class EventEngine:
    """
    Event engine distributes event object based on its type
    to those handlers registered.

    It also generates timer event by every interval seconds,
    which can be used for timing purpose.
    """

    def __init__(self, interval: int = 1) -> None:
        """
        Timer event is generated every 1 second by default, if
        interval not specified.
        """
        self._interval: int = interval
        self._queue: Queue = Queue()
        self._active: bool = False
        self._event_thread: Thread = Thread(target=self._process_events) # Consumer
        self._timer_thread: Thread = Thread(target=self._run_timer) # Producer
        self._handlers: defaultdict = defaultdict(list)
        self._general_handlers: list = []

    def _process_events(self) -> None:
        """Get event from queue and then process it."""
        while self._active:
            try:
                event: Event = self._queue.get(block=True, timeout=1)

                for handler in self._handlers.get(event.type, []):
                    handler(event)

                for handler in self._general_handlers:
                    handler(event)
            except Empty:
                pass


    def _run_timer(self) -> None:
        """
        Sleep by interval second(s) and then generate a timer event.
        """
        while self._active:
            sleep(self._interval)
            event: Event = Event(EVENT_TIMER)
            self.put(event)

    def start(self) -> None:
        """
        Start event engine to process events and generate timer events.
        """
        self._active = True
        self._event_thread.start()
        self._timer_thread.start()

    def stop(self) -> None:
        """
        Stop event engine.
        """
        self._active = False
        self._timer_thread.join()
        self._event_thread.join()

    def put(self, event: Event) -> None:
        """
        Put an event object into event queue.
        """
        logger.debug(f"EventEngine.put: {event.type}")
        self._queue.put(event)

    def register(self, type: str, handler: HandlerType) -> None:
        """
        Register a new handler function for a specific event type. Every
        function can only be registered once for each event type.
        """
        handler_list: list = self._handlers[type]
        if handler not in handler_list:
            handler_list.append(handler)

    def unregister(self, type: str, handler: HandlerType) -> None:
        """
        Unregister an existing handler function from event engine.
        """
        handler_list: list = self._handlers[type]
        if handler in handler_list:
            handler_list.remove(handler)

        if not handler_list:
            self._handlers.pop(type)

    def register_general(self, handler: HandlerType) -> None:
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def unregister_general(self, handler: HandlerType) -> None:
        if handler in self._general_handlers:
            self._general_handlers.remove(handler)
