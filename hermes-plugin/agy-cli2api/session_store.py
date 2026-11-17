"""Account-scoped in-memory session mapping for stateful agy --conversation resume."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field


def fingerprint(role: str, text: str) -> str:
    return hashlib.blake2s(
        f"{role}\n{text}".encode("utf-8", "replace"), digest_size=16
    ).hexdigest()


@dataclass
class _Session:
    conversation_id: str
    account: str
    sigs: list[str]
    last_used: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Plan:
    conversation_id: str | None
    prefix_len: int


class SessionStore:
    def __init__(self, max_sessions: int = 200) -> None:
        self._sessions: dict[str, _Session] = {}
        self._max = max(1, max_sessions)

    def lookup(self, sigs: list[str], account: str) -> Plan:
        """Find longest stored prefix matching `sigs` for `account`."""
        best: _Session | None = None
        for sess in self._sessions.values():
            if sess.account != account:
                continue
            n = len(sess.sigs)
            if n < len(sigs) and sigs[:n] == sess.sigs:
                if best is None or n > len(best.sigs):
                    best = sess
        if best is None:
            return Plan(conversation_id=None, prefix_len=0)
        best.last_used = time.time()
        return Plan(conversation_id=best.conversation_id, prefix_len=len(best.sigs))

    def remember(
        self,
        conversation_id: str,
        account: str,
        sigs: list[str],
        protected: set[str] | None = None,
    ) -> list[str]:
        self._sessions[conversation_id] = _Session(conversation_id, account, list(sigs))
        return self._evict(protected or set())

    def forget(self, conversation_id: str) -> None:
        self._sessions.pop(conversation_id, None)

    def forget_account(self, account: str) -> list[str]:
        """Evict only sessions owned by `account`, preserving all other accounts' caches."""
        victim_ids = [cid for cid, sess in self._sessions.items() if sess.account == account]
        for cid in victim_ids:
            self._sessions.pop(cid, None)
        return victim_ids

    def _evict(self, protected: set[str]) -> list[str]:
        evicted: list[str] = []
        while len(self._sessions) > self._max:
            candidates = [
                s for s in self._sessions.values() if s.conversation_id not in protected
            ]
            if not candidates:
                break
            oldest = min(candidates, key=lambda s: s.last_used)
            self._sessions.pop(oldest.conversation_id, None)
            evicted.append(oldest.conversation_id)
        return evicted
