"""Bus d'evenements : publication depuis n'importe quel fil, diffusion vers les clients WebSocket (asyncio)."""

from __future__ import annotations

import asyncio
import threading
import time


class Bus:
    def __init__(self, max_queue: int = 2000):
        self._subs: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.max_queue = max_queue
        self.seq = 0

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self.max_queue)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    @property
    def clients(self) -> int:
        return len(self._subs)

    def publish(self, evt: dict) -> None:
        with self._lock:
            self.seq += 1
            evt = {**evt, "seq": self.seq, "ts": round(time.time(), 3)}
            subs = list(self._subs)
        if not subs or self.loop is None or self.loop.is_closed():
            return

        def put():
            for q in subs:
                if q.full():   # client trop lent : on jette le plus ancien plutot que de bloquer l'agent
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(evt)
        try:
            self.loop.call_soon_threadsafe(put)
        except RuntimeError:
            pass
