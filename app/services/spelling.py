"""Проверка орфографии словарями Hunspell (через Enchant) — дешёвый фильтр ошибок OCR.

Безопасные автоисправления (визуально слово не меняется или меняется только регистр):
- смешение алфавитов: латинские буквы-двойники в русском слове и наоборот («OT» → «ОТ»,
  «Ресеivе» → «Receive») — если результат есть в словаре;
- случайный регистр («пОдраЗДелений» → «подразделений», «РЕпО» → «РЕПО») — если результат есть
  в словаре или так же написан в этом документе.
Остальные слова не из словаря — «подозрительные»: подсказки для ИИ и предупреждения. Слово,
одинаково написанное в документе 3+ раза, считается термином (ошибки OCR случайны).
Если Enchant или словари недоступны, проверка отключается и бот работает как раньше.
"""

import logging
import os
import re
from collections import Counter
from collections.abc import Iterable, Sequence

log = logging.getLogger(__name__)

DICT_TAGS = {"ru": "ru_RU", "en": "en_US"}
WORD = re.compile(r"[A-Za-zА-Яа-яЁё]+")
# Буквы-двойники латиницы и кириллицы (вид совпадает, коды разные).
LAT_TO_CYR = str.maketrans("ABCEHKMOPTXYaceopxyk", "АВСЕНКМОРТХУасеорхук")
CYR_TO_LAT = str.maketrans("АВСЕНКМОРТХУасеорхук", "ABCEHKMOPTXYaceopxyk")
# Для слов со смешанными алфавитами — ещё типичные путаницы OCR кириллица → латиница.
CYR_TO_LAT_OCR = str.maketrans("АВСЕНКМОРТХУасеорхукгпитьЕ", "ABCEHKMOPTXYaceopxykrnumbE")
LOOKALIKE_LAT = set("ABCEHKMOPTXYaceopxyk")
LOOKALIKE_CYR = set("АВСЕНКМОРТХУасеорхук")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-z]")
MAX_ABBREVIATION = 6
# Неизвестное слово, одинаково написанное в документе столько раз, — термин, а не ошибка OCR.
TERM_MIN_COUNT = 3


def _script_ratio(text: str) -> float:
    cyr, lat = len(CYRILLIC.findall(text)), len(LATIN.findall(text))
    return cyr / (cyr + lat) if cyr + lat else 0.0


def document_vocabulary(texts: Iterable[str]) -> Counter:
    vocab: Counter = Counter()
    for text in texts:
        vocab.update(WORD.findall(text))
    return vocab


class SpellChecker:
    def __init__(self, dicts: Sequence[object]):
        self._dicts = list(dicts)
        self._cache: dict[str, bool] = {}

    @classmethod
    def create(cls, languages: Iterable[str], dict_dir: str = "") -> "SpellChecker | None":
        """Словари для языков OCR; None, если Enchant или словари недоступны."""
        if dict_dir:
            # Enchant 2 ищет словари Hunspell в $ENCHANT_CONFIG_DIR/hunspell.
            os.environ["ENCHANT_CONFIG_DIR"] = dict_dir
        try:
            import enchant
        except Exception as exc:  # нет библиотеки libenchant
            log.warning("Spellcheck disabled: %s", type(exc).__name__)
            return None
        dicts = []
        for lang in languages:
            tag = DICT_TAGS.get(lang.lower())
            if not tag:
                continue
            try:
                dicts.append(enchant.Dict(tag))
            except Exception:
                log.warning("Spellcheck dictionary %s not found", tag)
        if not dicts:
            return None
        log.info("Spellcheck enabled: %s", ", ".join(d.tag for d in dicts))
        return cls(dicts)

    def known(self, word: str) -> bool:
        cached = self._cache.get(word)
        if cached is None:
            cached = any(d.check(word) for d in self._dicts)
            if len(self._cache) < 200_000:
                self._cache[word] = cached
        return cached

    def correct(self, word: str, cyrillic_context: bool, vocab: Counter | None = None) -> str | None:
        """Безопасное исправление слова или None."""
        has_cyr, has_lat = bool(CYRILLIC.search(word)), bool(LATIN.search(word))
        # 1. Слово целиком из двойников «чужого» алфавита: выглядит так же, поэтому переводим
        #    в алфавит текста даже если латинский вариант формально есть в словаре («OT» → «ОТ»).
        if has_lat and not has_cyr and cyrillic_context and set(word) <= LOOKALIKE_LAT:
            candidate = word.translate(LAT_TO_CYR)
            if self.known(candidate):
                return candidate
        if has_cyr and not has_lat and not cyrillic_context and set(word) <= LOOKALIKE_CYR:
            candidate = word.translate(CYR_TO_LAT)
            if self.known(candidate):
                return candidate
        if self.known(word):
            return None
        candidates: list[str] = []
        if has_cyr and has_lat:  # смешанные алфавиты: пробуем оба направления
            to_cyr, to_lat = word.translate(LAT_TO_CYR), word.translate(CYR_TO_LAT_OCR)
            candidates += [to_cyr, to_lat] if _script_ratio(word) >= 0.5 else [to_lat, to_cyr]
        base = candidates[:] or [word]
        for variant in base:  # случайный регистр
            if not (variant.islower() or variant.isupper() or variant.istitle()):
                candidates += [variant[0] + variant[1:].lower(), variant.lower(), variant.upper()]
        for candidate in candidates:
            if candidate != word and (self.known(candidate) or (vocab and vocab[candidate] >= 2)):
                return candidate
        return None

    def check_text(self, text: str, vocab: Counter | None = None) -> tuple[str, list[str]]:
        """Текст с безопасными исправлениями и список подозрительных слов (без повторов)."""
        cyrillic_context = _script_ratio(text) >= 0.5
        suspicious: list[str] = []
        seen: set[str] = set()

        def replace(match: re.Match) -> str:
            word = match.group(0)
            fixed = self.correct(word, cyrillic_context, vocab)
            if fixed:
                return fixed
            if word not in seen and self._suspicious(word, cyrillic_context, vocab):
                seen.add(word)
                suspicious.append(word)
            return word

        return WORD.sub(replace, text), suspicious

    def _suspicious(self, word: str, cyrillic_context: bool, vocab: Counter | None) -> bool:
        if len(word) < 3 or self.known(word):
            return False
        mixed = bool(CYRILLIC.search(word)) and bool(LATIN.search(word))
        lookalike_latin = bool(LATIN.search(word)) and set(word) <= LOOKALIKE_LAT and cyrillic_context
        if not mixed and vocab and vocab[word] >= TERM_MIN_COUNT:
            return False  # повторяющийся одинаково термин документа
        if word.isupper() and len(word) <= MAX_ABBREVIATION and not (lookalike_latin or mixed):
            return False  # аббревиатура: ЦБ, ПАО, УОРЦБ
        return True
