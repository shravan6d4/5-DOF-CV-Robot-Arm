"""One real, interactive shell (ConPTY via pywinpty), fanned out to every
browser tab connected to the dashboard.

There is one arm and one operator, so this is deliberately a single shared
session rather than a shell-per-tab: everyone watching the dashboard sees the
same terminal, the way they'd see the same physical console. A background
thread owns the PtyProcess and reads it continuously; each connected SSE
client gets its own Queue that the reader broadcasts every chunk into.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import List, Optional, Sequence, Tuple

import winpty

logger = logging.getLogger(__name__)

# Scrollback replayed to a tab that (re)connects mid-session. Not meant to be a
# full history -- xterm.js keeps its own once the client has caught up.
BACKLOG_CHARS = 8000


class TerminalSession:
    """Owns one PtyProcess and broadcasts its output to any number of subscribers."""

    def __init__(self, shell: Sequence[str], cwd: Optional[str] = None, rows: int = 24, cols: int = 80):
        self._shell = list(shell)
        self._cwd = cwd
        self._size: Tuple[int, int] = (rows, cols)
        self._lock = threading.Lock()
        self._subscribers: List["queue.Queue[str]"] = []
        self._backlog = ""
        self._stop = threading.Event()
        self._proc: Optional[winpty.PtyProcess] = None
        self._spawn()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _spawn(self) -> None:
        rows, cols = self._size
        self._proc = winpty.PtyProcess.spawn(self._shell, cwd=self._cwd, dimensions=(rows, cols))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._proc.read(4096)
            except EOFError:
                if self._stop.is_set():
                    return
                logger.warning("terminal session: shell exited, restarting")
                self._emit("\r\n\x1b[33m[shell exited -- restarting]\x1b[0m\r\n")
                try:
                    self._spawn()
                except OSError as e:
                    logger.error(f"terminal session: failed to respawn shell: {e}")
                    return
                continue
            except OSError:
                if self._stop.is_set():
                    return
                raise
            if chunk:
                self._emit(chunk)

    def _emit(self, chunk: str) -> None:
        with self._lock:
            self._backlog = (self._backlog + chunk)[-BACKLOG_CHARS:]
            for q in self._subscribers:
                q.put(chunk)

    def subscribe(self) -> "queue.Queue[str]":
        """Register a new listener; it immediately receives the current backlog."""
        q: "queue.Queue[str]" = queue.Queue()
        with self._lock:
            if self._backlog:
                q.put(self._backlog)
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def write(self, data: str) -> None:
        """Send raw input (keystrokes, including escape sequences) to the shell."""
        if self._proc is not None and self._proc.isalive():
            self._proc.write(data)

    def resize(self, rows: int, cols: int) -> None:
        self._size = (rows, cols)
        if self._proc is not None and self._proc.isalive():
            self._proc.setwinsize(rows, cols)

    def close(self) -> None:
        self._stop.set()
        if self._proc is not None:
            self._proc.close(force=True)
