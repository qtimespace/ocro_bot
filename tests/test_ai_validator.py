import pytest

from app.models import ValidationResult, WarningItem
from app.services.ai_validator import AIValidationError, AIValidator, apply_safeguards
from tests.conftest import FakeOpenAI, image_bytes, write

OK = {"corrected_text": "Привет, мир", "warnings": [{"fragment": "мир", "reason": "неясно"}], "needs_review": True}


def test_requires_api_key():
    with pytest.raises(AIValidationError):
        AIValidator("", "gpt-test")


async def test_text_mode_sends_only_text():
    client = FakeOpenAI([OK])
    result = await AIValidator("", "gpt-test", client=client).validate_text("Првиет, мир")
    assert result.corrected_text == "Привет, мир"
    assert result.needs_review
    request = client.requests[0]
    assert request["model"] == "gpt-test"
    assert request["response_format"] == {"type": "json_object"}
    assert "image_url" not in str(request["messages"])
    assert "do NOT have the source image" in request["messages"][0]["content"]


async def test_image_mode_sends_image_and_text(tmp_path):
    client = FakeOpenAI([OK])
    path = write(tmp_path, "p.png", image_bytes("PNG", size=(3000, 4000)))
    await AIValidator("", "gpt-test", client=client).validate_image(path, "Првиет, мир")
    content = client.requests[0]["messages"][1]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Првиет, мир" in content[0]["text"]


async def test_invalid_json_retried_then_fails():
    client = FakeOpenAI(["not json", '{"warnings": []}'])
    with pytest.raises(AIValidationError, match="формате"):
        await AIValidator("", "m", client=client).validate_text("text")
    assert len(client.requests) == 2


async def test_invalid_json_then_valid():
    client = FakeOpenAI(["not json", OK])
    result = await AIValidator("", "m", client=client).validate_text("Првиет, мир")
    assert result.corrected_text == "Привет, мир"


async def test_api_error_wrapped_without_details():
    client = FakeOpenAI([ConnectionError("secret-internal-detail")])
    with pytest.raises(AIValidationError) as exc_info:
        await AIValidator("", "m", client=client).validate_text("text")
    assert "secret-internal-detail" not in str(exc_info.value)


async def test_too_long_text_rejected():
    with pytest.raises(AIValidationError):
        await AIValidator("", "m", client=FakeOpenAI([OK])).validate_text("x" * 20_001)


def test_safeguard_empty_answer_keeps_ocr():
    result = apply_safeguards("Сумма 100 руб.", ValidationResult(corrected_text=""))
    assert result.corrected_text == "Сумма 100 руб."
    assert result.needs_review


def test_safeguard_large_change_keeps_ocr():
    ocr = "Договор поставки № 15 от 19.10.2026, сумма 1 000 000 рублей."
    result = apply_safeguards(ocr, ValidationResult(corrected_text=ocr + " Дополнительный выдуманный абзац." * 3))
    assert result.corrected_text == ocr
    assert result.needs_review


def test_safeguard_no_invention_on_empty_ocr():
    result = apply_safeguards("", ValidationResult(corrected_text="Выдуманный текст"))
    assert result.corrected_text == ""
    assert result.needs_review


def test_safeguard_passes_small_fix():
    fixed = ValidationResult(corrected_text="Привет, мир", warnings=[WarningItem(fragment="a", reason="b")])
    assert apply_safeguards("Првиет, мир", fixed) is fixed


async def test_temperature_dropped_when_model_rejects_it():
    class Picky(FakeOpenAI):
        async def _create(self, **kwargs):
            if "temperature" in kwargs:
                self.requests.append(kwargs)
                raise ValueError("Unsupported parameter: 'temperature' is not supported with this model.")
            return await super()._create(**kwargs)

    client = Picky([OK])
    validator = AIValidator("", "m", client=client)
    await validator.validate_text("Првиет, мир")
    await validator.validate_text("Првиет, мир")
    assert ["temperature" in r for r in client.requests] == [True, False, False]
