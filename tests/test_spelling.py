from app.services.ai_validator import AIValidator
from app.services.pipeline import Pipeline
from app.services.spelling import SpellChecker
from tests.conftest import FakeOcr, FakeOpenAI, image_bytes, make_settings, write

WORDS = {
    "от", "ОТ", "подразделений", "Подразделений", "договор", "Договор", "банк", "Банк", "БАНК", "сумма",
    "рублей", "организация", "organization", "total", "amount", "купли", "продажи",
}


class FakeDict:
    tag = "fake"

    def __init__(self, words):
        self.words = set(words)

    def check(self, word):
        return word in self.words


def speller():
    return SpellChecker([FakeDict(WORDS)])


def test_mixed_script_and_case_fixed_only_to_known_words():
    s = speller()
    assert s.correct("OT", cyrillic_context=True) == "ОТ"  # латиница-двойник в русском тексте
    assert s.correct("бaнк", cyrillic_context=True) == "банк"  # латинская «a» внутри слова
    assert s.correct("пОдраЗДелений", cyrillic_context=True) == "подразделений"
    assert s.correct("BAHK", cyrillic_context=True) is None  # «ВАНК» нет в словаре — не угадываем
    assert s.correct("Договор", cyrillic_context=True) is None  # и так верно


def test_check_text_fixes_and_collects_suspicious():
    text, suspicious = speller().check_text("Договор OT банка: сумма 100 рублей, пОдраЗДелений куплипродажи BAHK ЦБ")
    assert "ОТ" in text and "подразделений" in text
    assert suspicious == ["банка", "куплипродажи", "BAHK"]  # «ЦБ» — аббревиатура, не ошибка


def test_document_vocabulary_terms_and_case():
    from collections import Counter

    vocab = Counter({"Займодавец": 4, "РЕПО": 5})
    text, suspicious = speller().check_text("Займодавец заключает сделку РЕпО", vocab)
    assert "РЕПО" in text  # регистр исправлен по написанию в документе
    assert "Займодавец" not in suspicious  # повторяющийся термин — не ошибка
    assert "заключает" in suspicious and "сделку" in suspicious  # нет в маленьком тестовом словаре


def test_english_word_with_cyrillic_letters_fixed():
    s = SpellChecker([FakeDict({"Master", "Agreement"})])
    assert s.correct("Маsteг", cyrillic_context=True) == "Master"
    assert s.correct("Аgгеетепt", cyrillic_context=True) == "Agreement"
    # Кириллическая «Р» похожа на латинскую P, а не R: «Peceive» не слово — не исправляем.
    assert s.correct("Ресеivе", cyrillic_context=True) is None


def test_english_context_keeps_latin():
    text, suspicious = speller().check_text("total amount organization")
    assert text == "total amount organization" and suspicious == []


async def test_pipeline_uses_speller_for_warnings_and_ai_hints(tmp_path):
    settings = make_settings(tmp_path, OPENAI_API_KEY="sk-x")
    client = FakeOpenAI([{"corrected_text": "Договор ОТ банк куплипродажи"}])
    pipeline = Pipeline(settings, FakeOcr(["Договор OT банк куплипродажи"], confidence=0.3),
                        AIValidator("", "m", client=client), speller=speller())
    work = tmp_path / "work"
    work.mkdir()
    result = await pipeline.run(write(work, "in", image_bytes()), work, "text")
    page = result.pages[0]
    assert "ОТ" in page.text  # автоисправление до ИИ
    assert "куплипродажи" in client.requests[0]["messages"][1]["content"]  # подсказка для ИИ
    assert any("нет в словаре" in w and "куплипродажи" in w for w in result.warnings)
    assert not any("низкая уверенность" in w for w in result.warnings)  # шум заменён словарём


def test_create_returns_none_without_dictionaries():
    assert SpellChecker.create(["xx"]) is None
