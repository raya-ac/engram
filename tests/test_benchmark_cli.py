"""Runner safety checks without loading models or the real benchmark."""
import json
import sys

import pytest

from benchmarks.longmemeval import run_engram as runner


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / 'dataset.json'
    path.write_text(json.dumps([{
        'question_id': 'one', 'question_type': 'single-session-user',
        'question': 'which editor do I use?', 'answer': 'vim',
        'answer_session_ids': ['session'], 'haystack_sessions': [],
        'haystack_session_ids': [], 'haystack_dates': [],
    }]))
    return path


@pytest.mark.parametrize('existing', ['result', 'metadata', 'both'])
def test_fresh_run_refuses_existing_artifacts_before_models(dataset, tmp_path, monkeypatch, existing):
    output = tmp_path / 'result.jsonl'
    metadata = tmp_path / 'result.jsonl.metadata.json'
    before = {}
    for kind, path in [('result', output), ('metadata', metadata)]:
        if existing in (kind, 'both'):
            path.write_text('keep this existing artifact')
            before[path] = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must happen before inference')

    monkeypatch.setattr(runner, 'embed_query', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_engram', str(dataset), '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert all(path.read_bytes() == value for path, value in before.items())


def test_new_result_is_claimed_before_metadata_is_written(dataset, tmp_path, monkeypatch):
    output = tmp_path / 'result.jsonl'
    metadata = tmp_path / 'result.jsonl.metadata.json'
    write_metadata = runner.write_run_metadata

    def checked_write(path, value):
        assert output.is_file()
        assert output.read_bytes() == b''
        write_metadata(path, value)

    monkeypatch.setattr(runner, 'write_run_metadata', checked_write)
    monkeypatch.setattr(runner, 'embed_query', lambda *args: [0.0])
    monkeypatch.setattr(runner, 'engram_retrieve', lambda *args, **kwargs: [
        {'corpus_id': 'session', 'text': 'vim', 'timestamp': '2026-01-01'}
    ])
    monkeypatch.setattr(sys, 'argv', ['run_engram', str(dataset), '--output', str(output)])
    runner.main()
    assert json.loads(output.read_text())['question_id'] == 'one'
    assert json.loads(metadata.read_text())['config']['result_k'] == 5


def test_query_spelling_is_preserved_for_inference(monkeypatch):
    observed = []

    def stop_at_embedding(query, model):
        observed.append(query)
        raise RuntimeError('stop before model work')

    monkeypatch.setattr(runner, 'embed_query', stop_at_embedding)
    entry = {'haystack_sessions': [[{'role': 'user', 'content': 'opening my shop'}]],
             'haystack_session_ids': ['session'], 'haystack_dates': ['2026-01-01']}
    query = 'buisiness milestones'
    with pytest.raises(RuntimeError, match='stop before model work'):
        runner.engram_retrieve(query, entry, runner.Config())
    assert observed == [query]


def test_resume_of_missing_path_cannot_append_to_a_file_created_during_warmup(dataset, tmp_path, monkeypatch):
    output = tmp_path / 'result.jsonl'
    metadata = tmp_path / 'result.jsonl.metadata.json'

    def competing_run(*args):
        output.write_text('another run claimed this output')
        metadata.write_text('another configuration owns this result')
        return [0.0]

    monkeypatch.setattr(runner, 'embed_query', competing_run)
    monkeypatch.setattr(sys, 'argv', ['run_engram', str(dataset), '--output', str(output), '--resume'])
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_text() == 'another run claimed this output'
    assert metadata.read_text() == 'another configuration owns this result'


@pytest.mark.parametrize("overrides,expected_alpha,expected_floor", [
    ([], 0.25, 0.002),
    (["--fusion-alpha", "0.4"], 0.4, 0.002),
    (["--passage-floor", "0.01"], 0.25, 0.01),
    (["--fusion-alpha", "0.4", "--passage-floor", "0.01"], 0.4, 0.01),
    (["--fusion-alpha", "0", "--passage-floor", "0"], 0.0, 0.0),
])
def test_configured_fusion_and_passage_floor_survive_unless_overridden(
    dataset, tmp_path, monkeypatch, overrides, expected_alpha, expected_floor,
):
    output = tmp_path / "configured.jsonl"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("retrieval:\n  rerank_fusion_alpha: 0.25\n  rerank_passage_floor: 0.002\n")
    observed = []

    def retrieve(query, entry, config, **kwargs):
        observed.append((config.retrieval.rerank_fusion_alpha, config.retrieval.rerank_passage_floor))
        return [{"corpus_id": "session", "text": "vim", "timestamp": "2026-01-01"}]

    monkeypatch.setattr(runner, "embed_query", lambda *args: [0.0])
    monkeypatch.setattr(runner, "engram_retrieve", retrieve)
    monkeypatch.setattr(sys, "argv", [
        "run_engram", str(dataset), "--config", str(config_path), "--output", str(output), *overrides,
    ])
    runner.main()
    assert observed == [(expected_alpha, expected_floor)]
    recorded = json.loads(output.with_name(output.name + ".metadata.json").read_text())["config"]
    assert recorded["fusion_alpha"] == expected_alpha
    assert recorded["passage_confidence_floor"] == expected_floor


@pytest.mark.parametrize("floor", ["-0.1", "1.1", "nan"])
def test_invalid_cli_passage_floor_is_rejected_before_models(dataset, tmp_path, monkeypatch, floor):
    def forbidden(*args):
        pytest.fail("invalid passage policy must fail before inference")

    monkeypatch.setattr(runner, "embed_query", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "run_engram", str(dataset), "--output", str(tmp_path / "invalid.jsonl"), "--passage-floor", floor,
    ])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
