"""LLM response cache with optional verification metadata enrichment."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp (second precision, ``Z`` suffix)."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace(
        '+00:00', 'Z'
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    """Canonical sha256 of payload (sorted keys, compact separators)."""
    canonical = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class LLMCache:
    """File / sqlite / memory backed cache for LLM completions.

    Two-stage write protocol:
      * ``put_payload`` stores the raw response bytes once.
      * ``put`` enriches the same key with verification metadata (without
        rewriting the payload) so we can audit acceptance later.

    Every entry also tracks two audit fields that downstream aggregation
    relies on:

      * ``created_at``  - ISO-8601 UTC timestamp set at first insert; never
        rewritten on subsequent enrichment.
      * ``payload_hash`` - sha256 of canonical JSON of the stored payload;
        refreshed only when the payload itself is rewritten.
    """

    def __init__(self, cache_dir: str | Path, mode: str = 'memory') -> None:
        self._mode = mode
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        if mode == 'sqlite':
            self._db = self._dir / 'cache.sqlite'
            with sqlite3.connect(self._db) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS entries (
                      key TEXT PRIMARY KEY,
                      payload TEXT NOT NULL,
                      verification_result TEXT,
                      contract_version TEXT,
                      created_at TEXT,
                      payload_hash TEXT
                    )
                    """
                )
                # Forward-compatible migration for caches created before the
                # audit fields were added. ``ALTER TABLE`` is idempotent only
                # when guarded by a column-presence check.
                cols = {row[1] for row in conn.execute('PRAGMA table_info(entries)')}
                if 'created_at' not in cols:
                    conn.execute('ALTER TABLE entries ADD COLUMN created_at TEXT')
                if 'payload_hash' not in cols:
                    conn.execute('ALTER TABLE entries ADD COLUMN payload_hash TEXT')

    @staticmethod
    def make_key(prompt: str, contract_version: str, seed: int) -> str:
        body = f'{contract_version}|seed={seed}|{prompt}'
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            if self._mode == 'memory':
                entry = self._memory_cache.get(key)
                return dict(entry) if entry else None
            if self._mode == 'sqlite':
                with sqlite3.connect(self._db) as conn:
                    row = conn.execute(
                        'SELECT payload, verification_result, contract_version,'
                        ' created_at, payload_hash'
                        ' FROM entries WHERE key=?',
                        (key,),
                    ).fetchone()
                if row is None:
                    return None
                payload = json.loads(row[0])
                vr = json.loads(row[1]) if row[1] else None
                return {
                    'payload': payload,
                    'verification_result': vr,
                    'contract_version': row[2],
                    'created_at': row[3],
                    'payload_hash': row[4],
                }
        return None

    def put_payload(self, key: str, payload: dict[str, Any]) -> None:
        """Store raw LLM payload (no verification metadata yet).

        The payload may be rewritten on subsequent calls (e.g. retried
        generation), so ``payload_hash`` is refreshed each time. ``created_at``
        is preserved across rewrites.
        """
        new_hash = _payload_hash(payload)
        with self._lock:
            if self._mode == 'memory':
                if key in self._memory_cache:
                    self._memory_cache[key]['payload'] = payload
                    self._memory_cache[key]['payload_hash'] = new_hash
                else:
                    self._memory_cache[key] = {
                        'payload': payload,
                        'verification_result': None,
                        'contract_version': None,
                        'created_at': _utcnow_iso(),
                        'payload_hash': new_hash,
                    }
            elif self._mode == 'sqlite':
                with sqlite3.connect(self._db) as conn:
                    conn.execute(
                        'INSERT INTO entries(key, payload, created_at, payload_hash)'
                        ' VALUES(?, ?, ?, ?)\n'
                        'ON CONFLICT(key) DO UPDATE SET\n'
                        '  payload=excluded.payload,\n'
                        '  payload_hash=excluded.payload_hash,\n'
                        '  created_at=COALESCE(entries.created_at, excluded.created_at)',
                        (key, json.dumps(payload), _utcnow_iso(), new_hash),
                    )

    def put(
        self,
        key: str,
        payload: dict[str, Any],
        verification_result: dict[str, Any] | None,
        contract_version: str,
    ) -> None:
        """Enrich an existing entry with verification metadata.

        If the key has no payload yet, the supplied ``payload`` is written.
        When a payload already exists it is preserved (first-write-wins),
        and ``payload_hash`` matches whichever payload is actually stored.
        """
        candidate_hash = _payload_hash(payload)
        with self._lock:
            if self._mode == 'memory':
                if key not in self._memory_cache:
                    self._memory_cache[key] = {
                        'payload': payload,
                        'verification_result': verification_result,
                        'contract_version': contract_version,
                        'created_at': _utcnow_iso(),
                        'payload_hash': candidate_hash,
                    }
                else:
                    entry = self._memory_cache[key]
                    if entry.get('payload') is None:
                        entry['payload'] = payload
                        entry['payload_hash'] = candidate_hash
                    if entry.get('created_at') is None:
                        entry['created_at'] = _utcnow_iso()
                    if verification_result is not None:
                        entry['verification_result'] = verification_result
                    entry['contract_version'] = contract_version
            else:  # mode == 'sqlite'
                with sqlite3.connect(self._db) as conn:
                    conn.execute(
                        'INSERT INTO entries(key, payload, verification_result,'
                        ' contract_version, created_at, payload_hash)\n'
                        'VALUES(?, ?, ?, ?, ?, ?)\n'
                        'ON CONFLICT(key) DO UPDATE SET\n'
                        '  payload=COALESCE(entries.payload, excluded.payload),\n'
                        '  payload_hash=COALESCE(entries.payload_hash, excluded.payload_hash),\n'
                        '  verification_result=COALESCE(excluded.verification_result,'
                        ' entries.verification_result),\n'
                        '  contract_version=excluded.contract_version,\n'
                        '  created_at=COALESCE(entries.created_at, excluded.created_at)',
                        (
                            key,
                            json.dumps(payload),
                            json.dumps(verification_result) if verification_result else None,
                            contract_version,
                            _utcnow_iso(),
                            candidate_hash,
                        ),
                    )
