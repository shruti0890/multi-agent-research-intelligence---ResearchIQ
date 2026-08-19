"""
Tests for the quota-aware Gemini retry helper.
Validates that:
- GeminiQuotaExhaustedError is raised immediately on RESOURCE_EXHAUSTED (no retry)
- Transient errors (500, 503) are retried with backoff
- Successful responses pass through unchanged
"""
import sys
import os
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.agent_research import (
    _gemini_generate_with_retry,
    GeminiQuotaExhaustedError,
    _is_quota_exhausted,
)


# _is_quota_exhausted tests

def test_is_quota_exhausted_resource_exhausted():
    assert _is_quota_exhausted("429 RESOURCE_EXHAUSTED: You have exhausted your quota")

def test_is_quota_exhausted_daily_quota():
    assert _is_quota_exhausted("Exceeded daily quota for model")

def test_is_quota_exhausted_per_day():
    assert _is_quota_exhausted("requests_per_day limit exceeded")

def test_is_quota_exhausted_free_tier():
    assert _is_quota_exhausted("generate_content_free_tier limit reached")

def test_is_not_quota_exhausted_transient_429():
    assert not _is_quota_exhausted("429 Too Many Requests: retry after 60s")

def test_is_not_quota_exhausted_503():
    assert not _is_quota_exhausted("503 Service Unavailable")

def test_is_not_quota_exhausted_generic():
    assert not _is_quota_exhausted("500 Internal Server Error")


# _gemini_generate_with_retry quota path

def test_retry_raises_quota_error_immediately(monkeypatch):
    """When quota is exhausted, should raise GeminiQuotaExhaustedError without retrying."""
    call_count = {"n": 0}
    monkeypatch.setattr(time, "sleep", lambda x: None)

    class TrackingClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                call_count["n"] += 1
                raise Exception("RESOURCE_EXHAUSTED: daily quota exhausted")

    with pytest.raises(GeminiQuotaExhaustedError):
        _gemini_generate_with_retry(
            client=TrackingClient(),
            model="gemini-2.5-flash",
            prompt="test",
            config=None,
            max_retries=3,
            agent_label="Test",
        )

    assert call_count["n"] == 1, f"Expected 1 call (no retry on quota), got {call_count['n']}"


def test_retry_retries_on_transient_error(monkeypatch):
    """Transient 503 errors should be retried max_retries times before raising."""
    monkeypatch.setattr(time, "sleep", lambda x: None)
    call_count = {"n": 0}

    class TransientClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                call_count["n"] += 1
                raise Exception("503 Service Unavailable")

    with pytest.raises(Exception, match="503"):
        _gemini_generate_with_retry(
            client=TransientClient(),
            model="gemini-2.5-flash",
            prompt="test",
            config=None,
            max_retries=3,
            agent_label="Test",
        )

    assert call_count["n"] == 3, f"Expected 3 retries, got {call_count['n']}"


def test_retry_succeeds_on_first_attempt():
    class SuccessClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                class R:
                    text = "  hello world  "
                return R()

    result = _gemini_generate_with_retry(
        client=SuccessClient(),
        model="gemini-2.5-flash",
        prompt="test",
        config=None,
        max_retries=3,
        agent_label="Test",
    )
    assert result == "hello world"


def test_retry_succeeds_after_one_transient_failure(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda x: None)
    call_count = {"n": 0}

    class FlakeClient:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    raise Exception("500 Internal Server Error")
                class R:
                    text = "recovered"
                return R()

    result = _gemini_generate_with_retry(
        client=FlakeClient(),
        model="gemini-2.5-flash",
        prompt="test",
        config=None,
        max_retries=3,
        agent_label="Test",
    )
    assert result == "recovered"
    assert call_count["n"] == 2
