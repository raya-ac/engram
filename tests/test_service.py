"""Native JSONL framing and project/lifecycle contracts against isolated storage."""
from io import StringIO
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import pytest

from engram.config import Config
from engram.service import NativeService, MAX_REQUEST_BYTES, run_stdio
from engram.store import Memory, Store


@pytest.fixture
def native(tmp_path):
    config = Config(db_path=str(tmp_path / 'native.db'))
    config.ann.enabled = False
    config.ann.index_path = str(tmp_path / 'unused.index')
    store = Store(config)
    store.init_db()
    project = str(tmp_path / 'project')
    for ident, metadata, forgotten, status in (
        ('mine', {'project_path': project}, False, 'active'),
        ('other', {'project_path': str(tmp_path / 'other')}, False, 'active'),
        ('forgotten', {'project_path': project}, True, 'active'),
        ('inactive', {'project_path': project}, False, 'superseded'),
    ):
        store.save_memory(Memory(id=ident, content='Local test context ' + ident,
                                metadata=metadata, forgotten=forgotten, status=status,
                                access_count=2, importance=.7))
    service = NativeService(config)
    yield service, store, config, project
    service.close()
    store.close()


def call(service, operation, **params):
    return service.handle_request({'id': 'test', 'operation': operation, 'params': params})


def evidence_params(project):
    now = time.time()
    return dict(project_id=project, session_id='session', assumption_id='check', evidence_id='e1',
                outcome='supported', observed_at=now, expires_at=now+3600,
                observation={'actual': 'candidate hash matches expected'},
                provenance={'producer': 'native-test', 'check_type': 'checksum'},
                source_refs=['memory:mine'])


def test_discovery_truthful_scope_and_no_implicit_setup(tmp_path):
    service = NativeService(Config(db_path=str(tmp_path / 'missing' / 'new.db')))
    entries = call(service, 'operations')['result']['operations']
    schemas = {entry['name']: entry['inputSchema'] for entry in entries}
    assert 'query' in schemas['search']['properties']
    assert 'query' not in schemas['recall']['properties']
    assert 'project_id' not in schemas['search']['properties']
    assert 'project_id' not in schemas['dormant_inspect']['properties']
    assert 'remember' not in schemas
    assert call(service, 'status')['error']['code'] == 'operation_failed'
    assert not (tmp_path / 'missing').exists()


def test_scoped_recall_does_not_reinforce_or_cross_projects(native):
    service, store, config, project = native
    before = store.get_memory('mine')
    result = call(service, 'recall', project_id=project)['result']
    assert [m['id'] for m in result['memories']] == ['mine']
    assert 'not instructions' in result['boundary']
    after = store.get_memory('mine')
    assert (after.access_count, after.last_accessed, after.importance) == (before.access_count, before.last_accessed, before.importance)
    assert call(service, 'recall', project_id=project, query='test')['error']['code'] == 'invalid_params'
    assert call(service, 'recall', project_id='relative')['error']['code'] == 'invalid_params'
    assert call(service, 'search', query='test', project_id=project)['error']['code'] == 'invalid_params'


def test_explicit_checkpoints_and_evidence_survive_reopen(native):
    service, store, config, project = native
    assert call(service, 'session_checkpoint', project_id=project, task='release',
                summary='Client acceptance remains pending.', next_steps=['Check installed artifact'])['result']['status'] == 'saved'
    evidence = call(service, 'evidence_put', **evidence_params(project))['result']
    service.close()
    reopened = NativeService(config)
    try:
        resumed = call(reopened, 'session_resume', project_id=project, task='release')['result']
        assert resumed['checkpoints'][0]['summary'] == 'Client acceptance remains pending.'
        assert len(resumed['memories']) == 1  # inert evidence excluded from context
        item = call(reopened, 'evidence_get', project_id=project, id=evidence['id'])['result']
        assert item['state'] == 'supported' and item['verified_by_engram'] is False
        assert len(call(reopened, 'evidence_list', project_id=project)['result']) == 1
        assert call(reopened, 'resume', project_id=str(Path(project).parent / 'other'), task='release')['result']['checkpoints'] == []
        assert call(reopened, 'evidence_get', project_id=str(Path(project).parent / 'other'), id=evidence['id'])['result']['state'] == 'unknown'
        store.forget_memory('mine')
        assert call(reopened, 'evidence_get', project_id=project, id=evidence['id'])['result']['state'] == 'unknown'
        assert call(reopened, 'checkpoint', project_id=project, task='release', action='clear')['result']['status'] == 'cleared'
        assert call(reopened, 'resume', project_id=project, task='release')['result']['checkpoints'] == []
    finally:
        reopened.close()


def test_storewide_dormant_content_requires_explicit_inspect(native):
    service, store, config, project = native
    from engram.dormant import review
    review(config)
    now = time.time()
    store.conn.execute('''INSERT INTO dormant_recall_events
      (id,sequence,created_at,memory_id,outcome,candidate_count,relevance,bonus,
       dormant_days,overlap_count,query_term_count)
      VALUES ('event',1,?,'mine','selected',5,.82,.02,60,1,3)''', (now,))
    store.conn.execute('''INSERT INTO dormant_recall_state
      (memory_id,retrieved_at,cooldown_until) VALUES ('mine',?,?)''', (now, now+86400))
    store.conn.commit()
    before = store.get_memory('mine')
    rows = call(service, 'dormant_review')['result']
    assert rows[0]['shown_at'] is None and 'content' not in rows[0]
    assert 'error' in call(service, 'dormant_feedback', event_id='event', category='useful')
    assert call(service, 'dormant_inspect', event_id='event')['result']['content'] == before.content
    assert call(service, 'dormant_feedback', event_id='event', category='dismissed')['result']['feedback'] == 'dismissed'
    after = store.get_memory('mine')
    assert (before.access_count, before.last_accessed, before.importance) == (after.access_count, after.last_accessed, after.importance)
    store.forget_memory('mine')
    assert 'error' in call(service, 'dormant_inspect', event_id='event')
    assert 'error' in call(service, 'dormant_review', project_id=project)


def test_protocol_invalid_lines_recover_and_errors_are_sanitized(native, monkeypatch):
    service, store, config, project = native
    monkeypatch.setattr(NativeService, 'status', lambda self: (_ for _ in ()).throw(RuntimeError('SECRET database payload')))
    requests = ['not-json', '[]', '{"id":1,"operation":"status","params":{}}',
                '{"id":2,"operation":"operations","params":{}}', '{"id":3,"operation":"status","params":{"x":NaN}}',
                'x' * (MAX_REQUEST_BYTES + 10), '{"id":4,"operation":"remember","params":{}}']
    output = StringIO()
    run_stdio(config, StringIO('\n'.join(requests)+'\n'), output)
    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(responses) == 7
    assert responses[0]['error']['code'] == 'invalid_json'
    assert responses[1]['error']['code'] == 'invalid_request'
    assert responses[2]['id'] == 1 and responses[2]['error']['code'] == 'operation_failed'
    assert responses[3]['id'] == 2 and 'operations' in responses[3]['result']
    assert responses[4]['error']['code'] == 'invalid_json'
    assert responses[5]['error']['code'] == 'request_too_large'
    assert responses[6]['error']['code'] == 'unknown_operation'
    assert 'SECRET' not in output.getvalue()


def test_protocol_keeps_accidental_diagnostics_off_stdout(native, monkeypatch, capsys):
    service, store, config, project = native
    def noisy(_self):
        print('diagnostic-only')
        return {'healthy': True}
    monkeypatch.setattr(NativeService, 'status', noisy)
    output = StringIO()
    run_stdio(config, StringIO('{"id":1,"operation":"status"}\n'), output)
    assert json.loads(output.getvalue()) == {'id': 1, 'result': {'healthy': True}}
    assert capsys.readouterr().err == 'diagnostic-only\n'


def test_actual_cli_persistent_jsonl_and_restart(native, tmp_path):
    service, store, config, project = native
    import yaml
    path = tmp_path / 'native.yaml'
    path.write_text(yaml.safe_dump({'db_path': config.db_path, 'ann': {'enabled': False}}))
    command = [sys.executable, '-m', 'engram', '--config', str(path), 'api']
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    def request(operation, **params):
        proc.stdin.write(json.dumps({'id': operation, 'operation': operation, 'params': params})+'\n')
        proc.stdin.flush()
        assert selector.select(20), 'Native API did not flush a response before EOF'
        return json.loads(proc.stdout.readline())
    try:
        first = request('status')['result']
        assert first['pid'] == proc.pid
        assert request('session_checkpoint', project_id=project, task='native', summary='Resume through JSONL')['result']['status'] == 'saved'
        saved = request('evidence_put', **evidence_params(project))['result']
        assert request('evidence_get', project_id=project, id=saved['id'])['result']['eligible'] is True
        assert request('status')['result']['pid'] == first['pid']
        proc.stdin.close()
        assert proc.wait(timeout=10) == 0
        assert not proc.stdout.read().strip()
    finally:
        selector.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    payload = {'id': 'resumed', 'operation': 'session_resume', 'params': {'project_id': project, 'task': 'native'}}
    result = subprocess.run(command, input=json.dumps(payload)+'\n', text=True, capture_output=True, timeout=20, check=True)
    assert json.loads(result.stdout)['result']['checkpoints'][0]['summary'] == 'Resume through JSONL'
    assert not result.stderr.strip()


def test_startup_discovery_and_context_do_not_import_models_or_mcp(native):
    service, store, config, project = native
    code = '''import sys
from engram.config import Config
from engram.service import NativeService
s=NativeService(Config(db_path=sys.argv[1]))
for operation,params in [('operations',{}),('status',{}),('recall',{'project_id':sys.argv[2]})]:
 result=s.handle_request({'id':1,'operation':operation,'params':params})
 assert 'error' not in result, result
assert not any(name.startswith(('engram.mcp_server','engram.embeddings','engram.retrieval','sentence_transformers','mlx')) for name in sys.modules)
s.close()
'''
    subprocess.run([sys.executable, '-c', code, config.db_path, project], check=True, capture_output=True, timeout=20)


def test_invalid_unicode_frames_do_not_kill_actual_worker(native, tmp_path):
    service, store, config, project = native
    import yaml
    path = tmp_path / 'unicode.yaml'
    path.write_text(yaml.safe_dump({'db_path': config.db_path, 'ann': {'enabled': False}}))
    frames = (b'{"id":"\\ud800","operation":"status"}\n'
              b'{"id":"\xff","operation":"status"}\n'
              b'{"id":"healthy","operation":"status"}\n')
    result = subprocess.run([sys.executable, '-m', 'engram', '--config', str(path), 'api'],
                            input=frames, capture_output=True, timeout=20, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert rows[0]['id'] == '\ud800' and 'result' in rows[0]
    assert rows[1]['error']['code'] == 'invalid_json'
    assert rows[2]['id'] == 'healthy' and 'result' in rows[2]
    assert not result.stderr


def test_persistent_search_keeps_models_warm(native, monkeypatch):
    service, store, config, project = native
    from engram import embeddings, retrieval
    calls = []
    monkeypatch.setattr(embeddings, 'set_backend', lambda value: calls.append(('backend', value)))
    monkeypatch.setattr(embeddings, 'set_default_model', lambda value: calls.append(('model', value)))
    monkeypatch.setattr(retrieval, 'search', lambda *args, **kwargs: calls.append(('search', kwargs['top_k'])) or [])
    assert call(service, 'search', query='release artifact')['result'] == []
    assert call(service, 'search', query='restore checkpoint')['result'] == []
    assert [name for name, _ in calls] == ['backend', 'model', 'search', 'search']
