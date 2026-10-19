import pytest
from pydantic import ValidationError

from app.models import OcrLine, OcrResult, ValidationResult


def test_validation_result_parses():
    result = ValidationResult.model_validate({
        "corrected_text": "Hello\nПривет",
        "warnings": [{"fragment": "123", "reason": "Check number"}],
        "needs_review": True,
    })
    assert result.needs_review is True
    assert result.warnings[0].fragment == "123"


def test_validation_result_defaults():
    result = ValidationResult(corrected_text="test")
    assert result.warnings == []
    assert result.needs_review is False


def test_validation_result_requires_text():
    with pytest.raises(ValidationError):
        ValidationResult.model_validate({"warnings": []})


def test_validation_result_rejects_wrong_types():
    with pytest.raises(ValidationError):
        ValidationResult.model_validate({"corrected_text": "x", "needs_review": "maybe"})


def test_string_warnings_coerced_and_extra_ignored():
    result = ValidationResult.model_validate(
        {"corrected_text": "x", "warnings": ["unclear date"], "warnings_null": None, "extra": 1}
    )
    assert result.warnings[0].reason == "unclear date"


def test_ocr_result_text_joins_rows():
    result = OcrResult(rows=[[OcrLine("A"), OcrLine("B")], [OcrLine("C")], [OcrLine("")]])
    assert result.text == "A B\nC"
    assert len(result.fragments) == 4
