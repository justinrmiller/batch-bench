from __future__ import annotations

import logging

import pytest
from torch.distributed.elastic.multiprocessing import redirects

import batch_bench  # noqa: F401 - importing the package installs the filter


def test_macos_redirect_note_is_dropped(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    monkeypatch.setattr(redirects, "IS_MACOS", True)
    with caplog.at_level(logging.WARNING):
        assert redirects.get_libc() is None
    assert caplog.records == []


def test_other_messages_from_that_logger_still_appear(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING):
        redirects.logger.warning("something else")
    assert [record.getMessage() for record in caplog.records] == ["something else"]
