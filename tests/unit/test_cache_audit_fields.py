"""Regression: cache entries must carry ``created_at`` and ``payload_hash``.

``created_at`` is immutable after first write; ``payload_hash`` tracks the
currently-stored payload (refreshed on ``put_payload`` rewrites, preserved on
``put`` enrichment).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

from gread.llm.cache import LLMCache, _payload_hash

ISO_RE = re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$')


def _expected_hash(payload: dict) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def test_payload_hash_helper_canonical():
    a = {'a': 1, 'b': 2}
    b = {'b': 2, 'a': 1}
    # Order-independent canonical form.
    assert _payload_hash(a) == _payload_hash(b) == _expected_hash(a)


def test_memory_put_payload_writes_audit_fields(tmp_path: Path):
    cache = LLMCache(str(tmp_path), mode='memory')
    cache.put_payload('k', {'foo': 'bar'})
    entry = cache.get('k')
    assert entry is not None
    assert ISO_RE.match(entry['created_at'])
    assert entry['payload_hash'] == _expected_hash({'foo': 'bar'})


def test_memory_put_payload_rewrite_refreshes_hash_keeps_created_at(tmp_path: Path):
    cache = LLMCache(str(tmp_path), mode='memory')
    cache.put_payload('k', {'v': 1})
    first = cache.get('k')
    assert first is not None
    time.sleep(1.1)  # ensure clock advances past second-precision boundary
    cache.put_payload('k', {'v': 2})
    second = cache.get('k')
    assert second is not None
    assert second['created_at'] == first['created_at']  # immutable
    assert second['payload_hash'] == _expected_hash({'v': 2})  # refreshed
    assert second['payload_hash'] != first['payload_hash']


def test_memory_put_then_put_payload_preserves_created_at(tmp_path: Path):
    cache = LLMCache(str(tmp_path), mode='memory')
    cache.put('k', {'a': 1}, {'ok': True}, 'gread_v1')
    first = cache.get('k')
    assert first is not None
    assert first['payload_hash'] == _expected_hash({'a': 1})
    cache.put_payload('k', {'a': 1})
    second = cache.get('k')
    assert second is not None
    assert second['created_at'] == first['created_at']
    assert second['verification_result'] == {'ok': True}


def test_memory_put_after_put_payload_preserves_payload_hash(tmp_path: Path):
    cache = LLMCache(str(tmp_path), mode='memory')
    cache.put_payload('k', {'a': 1})
    first = cache.get('k')
    assert first is not None
    # Verifier wins, but its candidate payload is ignored when one already exists.
    cache.put('k', {'a': 99}, {'ok': True}, 'gread_v1')
    second = cache.get('k')
    assert second is not None
    assert second['payload'] == {'a': 1}  # first-write-wins
    assert second['payload_hash'] == first['payload_hash']
    assert second['verification_result'] == {'ok': True}
    assert second['contract_version'] == 'gread_v1'


def test_sqlite_audit_fields_round_trip(tmp_path: Path):
    cache = LLMCache(str(tmp_path), mode='sqlite')
    cache.put('k', {'foo': 1}, {'ok': True}, 'gread_v1')
    entry = cache.get('k')
    assert entry is not None
    assert ISO_RE.match(entry['created_at'])
    assert entry['payload_hash'] == _expected_hash({'foo': 1})


def test_sqlite_legacy_db_migrates_audit_columns(tmp_path: Path):
    """A pre-audit-field DB must auto-add the new columns on open."""
    db = tmp_path / 'cache.sqlite'
    with sqlite3.connect(db) as conn:
        conn.execute(
            'CREATE TABLE entries ('
            ' key TEXT PRIMARY KEY,'
            ' payload TEXT NOT NULL,'
            ' verification_result TEXT,'
            ' contract_version TEXT)'
        )
        conn.execute(
            'INSERT INTO entries(key, payload) VALUES(?, ?)',
            ('legacy', json.dumps({'old': True})),
        )
    cache = LLMCache(str(tmp_path), mode='sqlite')
    legacy = cache.get('legacy')
    assert legacy is not None
    # Columns exist (migration ran) but legacy rows have NULL audit fields.
    assert legacy['created_at'] is None
    assert legacy['payload_hash'] is None
    # New writes still populate them.
    cache.put('fresh', {'x': 1}, {'ok': True}, 'gread_v1')
    fresh = cache.get('fresh')
    assert fresh is not None
    assert ISO_RE.match(fresh['created_at'])
    assert fresh['payload_hash'] == _expected_hash({'x': 1})
