import pytest
from pydantic import ValidationError

from tests.conftest import make_settings


def test_defaults(tmp_path):
    s = make_settings(tmp_path)
    assert s.ai_validation_mode == "off"
    assert s.docx_mode == "facsimile"
    assert s.languages == ["ru", "en"]
    assert s.max_upload_bytes == 20 * 1024 * 1024
    assert s.max_pdf_pages == 30
    assert s.max_image_pixels == 40_000_000
    assert s.ai_available is False


def test_allowlist_parsing(tmp_path):
    s = make_settings(tmp_path, ALLOWED_TELEGRAM_USER_IDS=" 10, 20 ,,30")
    assert s.allowed_ids == {10, 20, 30}
    assert s.is_allowed(20)
    assert not s.is_allowed(99)
    assert not s.is_allowed(None)


def test_empty_allowlist_allows_everyone(tmp_path):
    s = make_settings(tmp_path)
    assert s.is_allowed(12345)


def test_invalid_allowlist_rejected(tmp_path):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, ALLOWED_TELEGRAM_USER_IDS="10,abc")


@pytest.mark.parametrize("value", ["IMAGE", " text "])
def test_mode_normalized(tmp_path, value):
    assert make_settings(tmp_path, AI_VALIDATION_MODE=value).ai_validation_mode == value.strip().lower()


@pytest.mark.parametrize("key,value", [("AI_VALIDATION_MODE", "optional"), ("DOCX_MODE", "pdf"), ("MAX_UPLOAD_MB", "0")])
def test_invalid_values_rejected(tmp_path, key, value):
    with pytest.raises(ValidationError):
        make_settings(tmp_path, **{key: value})


def test_secrets_not_in_repr(tmp_path):
    s = make_settings(tmp_path, OPENAI_API_KEY="sk-secret-value-123")
    assert s.ai_available
    assert "sk-secret-value-123" not in repr(s)
    assert "sk-secret-value-123" not in str(s.model_dump())


def test_env_example_lists_required_variables():
    from pathlib import Path

    text = (Path(__file__).parent.parent / ".env.example").read_text(encoding="utf-8")
    for name in ("TELEGRAM_BOT_TOKEN", "OPENAI_API_KEY", "OPENAI_MODEL", "AI_VALIDATION_MODE", "DOCX_MODE",
                 "OCR_LANGUAGES", "MAX_UPLOAD_MB", "MAX_PDF_PAGES", "MAX_IMAGE_PIXELS", "TEMP_DIR", "LOG_LEVEL",
                 "ALLOWED_TELEGRAM_USER_IDS"):
        assert f"{name}=" in text


def test_openrouter_used_when_openai_key_missing(tmp_path):
    s = make_settings(tmp_path, OPENROUTER_API_KEY="sk-or-test", OPENROUTER_MODEL="openai/gpt-x")
    assert s.ai_available
    assert s.resolved_ai_provider == "openrouter"
    assert s.ai_api_key == "sk-or-test"
    assert s.ai_model == "openai/gpt-x"
    assert s.ai_base_url == "https://openrouter.ai/api/v1"


def test_openai_preferred_in_auto_and_explicit_provider(tmp_path):
    s = make_settings(tmp_path, OPENAI_API_KEY="sk-oa", OPENROUTER_API_KEY="sk-or")
    assert s.resolved_ai_provider == "openai" and s.ai_base_url is None
    s = make_settings(tmp_path, OPENAI_API_KEY="sk-oa", OPENROUTER_API_KEY="sk-or", AI_PROVIDER="openrouter")
    assert s.ai_api_key == "sk-or"
    assert not make_settings(tmp_path, OPENROUTER_API_KEY="sk-or", AI_PROVIDER="openai").ai_available


def test_env_names_case_insensitive(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("OpenRouter_API_KEY=sk-or-file\nOpenRouter_MODEL=openai/gpt-y\n", encoding="utf-8")
    from app.config import Settings

    s = Settings(_env_file=str(env), TEMP_DIR=str(tmp_path))
    assert s.ai_api_key == "sk-or-file"
    assert s.ai_model == "openai/gpt-y"
