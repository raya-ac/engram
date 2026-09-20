"""Only the known non-persistent BGE buffer report should disappear."""
import logging

import pytest

from engram.embeddings import _BGEPositionBufferFilter


REPORT = """XLMRobertaForSequenceClassification LOAD REPORT from: BAAI/bge-reranker-base
Key                             | Status     |  |
--------------------------------+------------+--+--
roberta.embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED:\tcan be ignored when loading from different task/architecture; not ok if you expect identical arch.
"""


def record(message=REPORT, *, level=logging.WARNING, name="transformers.modeling_utils", **kwargs):
    return logging.LogRecord(name, level, __file__, 1, message, (), kwargs.get("exc_info"),
                             sinfo=kwargs.get("stack_info"))


@pytest.mark.parametrize("message", [
    REPORT,
    REPORT.replace("XLMRobertaForSequenceClassification LOAD REPORT", "\x1b[1mXLMRobertaForSequenceClassification LOAD REPORT\x1b[0m")
          .replace("UNEXPECTED", "\x1b[38;5;208mUNEXPECTED\x1b[0m"),
    REPORT.replace("|  |", "| Details", 1).replace("|  |", "|", 1),
    REPORT.replace("XLMRobertaForSequenceClassification", "BertModel")
          .replace("BAAI/bge-reranker-base", "BAAI/bge-small-en-v1.5")
          .replace("roberta.embeddings.position_ids", "embeddings.position_ids"),
])
@pytest.mark.parametrize("logger_name", _BGEPositionBufferFilter._loggers)
def test_known_buffer_only_report_is_quiet(message, logger_name):
    assert not _BGEPositionBufferFilter().filter(record(message, name=logger_name))


@pytest.mark.parametrize("message", [
    REPORT.replace("roberta.embeddings.position_ids", "classifier.weight"),
    REPORT.replace("roberta.embeddings.position_ids", "roberta.embeddings.position_embeddings.weight"),
    REPORT.replace("UNEXPECTED", "MISSING"),
    REPORT.replace("UNEXPECTED", "MISMATCH"),
    REPORT.replace("BAAI/bge-reranker-base", "another/model"),
    REPORT.replace("XLMRobertaForSequenceClassification", "AnotherArchitecture"),
    REPORT.replace("\n\nNotes:", "\nclassifier.weight | MISSING | |\n\nNotes:"),
    REPORT + "additional diagnostic context\n",
    REPORT.replace("not ok if you expect identical arch.", "new upstream diagnostic"),
    "roberta.embeddings.position_ids failed to initialize",
])
def test_changed_or_actionable_report_is_preserved(message):
    assert _BGEPositionBufferFilter().filter(record(message))


@pytest.mark.parametrize("options", [
    {"level": logging.ERROR},
    {"name": "another.library"},
    {"exc_info": (RuntimeError, RuntimeError("load failed"), None)},
    {"stack_info": "stack trace"},
])
def test_exception_context_and_other_loggers_are_preserved(options):
    assert _BGEPositionBufferFilter().filter(record(**options))


@pytest.mark.parametrize("model_class,model_id,key", [
    ("XLMRobertaForSequenceClassification", "BAAI/bge-reranker-base", "roberta.embeddings.position_ids"),
    ("BertModel", "BAAI/bge-small-en-v1.5", "embeddings.position_ids"),
])
def test_actual_transformers_report_and_missing_weights(caplog, model_class, model_id, key):
    loading = pytest.importorskip("transformers.utils.loading_report")
    logger = logging.getLogger("transformers.modeling_utils")
    model = type(model_class, (), {})()
    info = loading.LoadStateDictInfo(set(), {key}, set(), [], {})
    # Transformers disables propagation to the root logger by default.
    logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            loading.log_state_dict_report(model, model_id, False, info, logger)
            assert not caplog.records
            info.missing_keys.add("classifier.weight")
            loading.log_state_dict_report(model, model_id, False, info, logger)
    finally:
        logger.removeHandler(caplog.handler)
    assert len(caplog.records) == 1
    assert "classifier.weight" in caplog.records[0].getMessage()
    assert "MISSING" in caplog.records[0].getMessage()


def test_model_load_failures_still_raise():
    loading = pytest.importorskip("transformers.utils.loading_report")
    info = loading.LoadStateDictInfo(set(), {"roberta.embeddings.position_ids"}, set(),
                                    ["checkpoint could not be loaded"], {})
    model = type("XLMRobertaForSequenceClassification", (), {})()
    with pytest.raises(RuntimeError, match="checkpoint could not be loaded"):
        loading.log_state_dict_report(model, "BAAI/bge-reranker-base", False, info)
