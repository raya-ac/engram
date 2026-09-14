"""Authenticated review uses separate exposure/feedback, never memory reinforcement."""
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from engram.config import Config
from engram.dormant import review
from engram.store import Memory
from engram.web.app import create_app


@pytest.fixture
def web(tmp_path):
    cfg = Config()
    cfg.db_path = str(tmp_path / 'web.db')
    cfg.ann.enabled = False
    cfg.ann.index_path = str(tmp_path / 'unused.index')
    cfg.dormant_recall.mode = 'shadow'
    cfg.web.auth_token = 'isolated-test-token'
    # No model/index warmup: these tests exercise the actual web/storage boundary.
    with patch('engram.web.app.threading.Thread.start'):
        app = create_app(cfg)
    db = app.state.store
    now = time.time()
    db.save_memory(Memory(id='rare', content='Verify the installed release hash.',
                          importance=.7, access_count=2, created_at=now-90*86400,
                          last_accessed=now-60*86400))
    review(cfg)
    db.conn.execute('''INSERT INTO dormant_recall_events
      (id,sequence,created_at,memory_id,outcome,candidate_count,relevance,bonus,
       dormant_days,overlap_count,query_term_count)
      VALUES ('event',1,?,'rare','selected',5,.82,.02,60,1,3)''', (now,))
    db.conn.execute('''INSERT INTO dormant_recall_state
      (memory_id,retrieved_at,cooldown_until) VALUES ('rare',?,?)''', (now,now+86400))
    db.conn.commit()
    with TestClient(app) as client:
        yield client, cfg, db
    db.close()


def headers():
    return {'Authorization': 'Bearer isolated-test-token'}


def counters(db):
    m = db.get_memory('rare')
    return m.access_count, m.last_accessed, m.importance


def test_review_is_metadata_only_and_protected(web):
    client, cfg, db = web
    assert client.get('/api/dormant').status_code == 401
    for path in ('inspect','feedback'):
        assert client.post('/api/dormant/event/'+path, json={}).status_code == 401
    before = counters(db)
    response = client.get('/api/dormant', headers=headers())
    assert response.status_code == 200
    data = response.json()
    assert data['mode'] == 'shadow'
    assert data['events'][0]['shown_at'] is None
    assert 'content' not in data['events'][0]
    assert 'query_text' not in data['events'][0]
    assert counters(db) == before
    assert client.get('/api/dormant?limit=1000',headers=headers()).status_code == 422


def test_explicit_inspection_and_feedback_do_not_reinforce(web):
    client, cfg, db = web
    before = counters(db)
    assert client.post('/api/dormant/event/feedback',headers=headers(),
                       json={'category':'useful'}).status_code == 409
    response = client.post('/api/dormant/event/inspect',headers=headers(),json={})
    assert response.status_code == 200
    assert response.json()['content'] == 'Verify the installed release hash.'
    assert client.get('/api/dormant',headers=headers()).json()['events'][0]['shown_at']
    assert counters(db) == before
    response = client.post('/api/dormant/event/feedback',headers=headers(),json={'category':'dismissed'})
    assert response.json()['feedback'] == 'dismissed'
    assert counters(db) == before
    assert client.post('/api/dormant/event/feedback',headers=headers(),json={'category':'useful'}).status_code == 409
    assert client.post('/api/dormant/event/feedback',headers=headers(),json={'category':'silence'}).status_code == 422


def test_archived_candidate_cannot_be_opened(web):
    client, cfg, db = web
    db.forget_memory('rare')
    assert client.post('/api/dormant/event/inspect',headers=headers(),json={}).status_code == 409


def test_all_views_and_local_assets_render(web):
    client, cfg, db = web
    response = client.get('/',headers=headers())
    assert response.status_code == 200
    html = response.text
    for name in ('neural','search','continuity','intelligence','memories','entities',
                 'timeline','remember','analytics','context','ingest','health','dedup',
                 'heatmap','cognition','bridges','drift','patterns','dormant'):
        assert f'id="view-{name}"' in html
    assert 'https://fonts.googleapis.com' not in html
    assert 'unpkg.com' not in html
    assert 'Open navigation' in html
    assert 'Memory inspector and activity' in html
