"""Credentials live in the environment, never in config.

A key in configs/ would be hashed into the run id, copied verbatim into
runs/*/config.resolved.yaml, and rendered on a page that prints the config
hash. The tests here are about that boundary, not about dotenv parsing.
"""
from __future__ import annotations

from kbeval.config import load_config
from kbeval.env import load_dotenv


def test_values_are_read_and_names_reported(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('KBEVAL_TEST_TOKEN = "s3cret"\n', encoding="utf-8")
    monkeypatch.delenv("KBEVAL_TEST_TOKEN", raising=False)
    assert "KBEVAL_TEST_TOKEN" in load_dotenv(p)
    import os
    assert os.environ["KBEVAL_TEST_TOKEN"] == "s3cret"


def test_a_shell_export_wins_over_the_file(tmp_path, monkeypatch):
    """An export is a deliberate act; a file must not silently override it."""
    p = tmp_path / ".env"
    p.write_text("KBEVAL_TEST_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("KBEVAL_TEST_TOKEN", "from_shell")
    load_dotenv(p)
    import os
    assert os.environ["KBEVAL_TEST_TOKEN"] == "from_shell"


def test_the_bedrock_alias_is_populated(tmp_path, monkeypatch):
    """botocore reads AWS_BEARER_TOKEN_BEDROCK; the console calls the same
    string BEDROCK_API_KEY."""
    p = tmp_path / ".env"
    p.write_text("BEDROCK_API_KEY=abc\n", encoding="utf-8")
    monkeypatch.delenv("BEDROCK_API_KEY", raising=False)
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    names = load_dotenv(p)
    assert {"BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"} <= set(names)


def test_comments_blanks_and_junk_are_skipped(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\n\nNOT_AN_ASSIGNMENT\nEMPTY=\n", encoding="utf-8")
    assert load_dotenv(p) == []


def test_a_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope") == []


def test_no_credential_can_reach_the_config(base_yaml):
    """The boundary itself: nothing secret-shaped is a config field, so a key
    cannot be hashed into a run id even by mistake."""
    cfg = load_config([base_yaml])
    flat = cfg.canonical().lower()
    # not "token": context_token_budget and max_tokens are legitimate settings
    for word in ("api_key", "apikey", "bearer", "secret", "password", "credential", "absk"):
        assert word not in flat


def test_the_resolved_dump_carries_no_secrets(base_yaml, tmp_path):
    from kbeval.config import dump_resolved
    out = tmp_path / "config.resolved.yaml"
    dump_resolved(load_config([base_yaml]), out)
    body = out.read_text(encoding="utf-8").lower()
    assert "absk" not in body and "bearer" not in body
