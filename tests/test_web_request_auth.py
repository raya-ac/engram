"""Execute the workspace's request wrapper without a browser or network."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_workspace_token_stays_on_its_origin_and_preserves_request_options():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute the frontend request wrapper")
    template = (Path(__file__).resolve().parents[1] / "engram/web/templates/index.html").read_text()
    function = "async function fetchAPI(" + template.split("async function fetchAPI(", 1)[1].split("\nasync function fetchJSON", 1)[0]
    script = r"""
const assert = require('node:assert/strict');
const calls = [];
global.window = {location: new URL('http://127.0.0.1:8849/?token=demo%2Btoken#search')};
global.fetch = async (url, options) => { calls.push({url, options}); return {ok: true}; };
FUNCTION
(async () => {
  await fetchAPI('/api/stats');
  assert.equal(calls[0].options.headers.get('Authorization'), 'Bearer demo+token');
  const signal = new AbortController().signal;
  const options = {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{"dry_run":true}', signal};
  await fetchAPI('/api/drift/fix', options);
  assert.equal(calls[1].options.headers.get('Content-Type'), 'application/json');
  assert.equal(calls[1].options.headers.get('Authorization'), 'Bearer demo+token');
  assert.equal(calls[1].options.method, 'POST');
  assert.equal(calls[1].options.body, options.body);
  assert.equal(calls[1].options.signal, signal);
  assert.deepEqual(options.headers, {'Content-Type': 'application/json'});
  await fetchAPI('http://127.0.0.1:8850/api/stats');
  await fetchAPI('https://example.invalid/api/stats');
  assert.equal(calls[2].options.headers.has('Authorization'), false);
  assert.equal(calls[3].options.headers.has('Authorization'), false);
  window.location = new URL('http://127.0.0.1:8849/');
  await fetchAPI('/api/stats');
  assert.equal(calls[4].options.headers.has('Authorization'), false);
})().catch(error => { console.error(error); process.exitCode = 1; });
""".replace("FUNCTION", function)
    completed = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
