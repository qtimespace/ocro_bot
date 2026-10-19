from app.models import OcrLine, OcrResult
from app.services.page_order import detect_page_number, has_conflicting_numbers, order_by_page_numbers

H = 1000


def ocr(*items: tuple[str, float]) -> OcrResult:
    """items: (текст, вертикальный центр фрагмента)."""
    return OcrResult(rows=[[OcrLine(text, box=(100, y - 10, 300, y + 10))] for text, y in items])


def test_bare_number_in_footer():
    assert detect_page_number(ocr(("Текст договора", 300), ("- 7 -", 970)), H) == 7
    assert detect_page_number(ocr(("12", 960)), H) == 12


def test_explicit_formats():
    assert detect_page_number(ocr(("Страница 3 из 10", 960)), H) == 3
    assert detect_page_number(ocr(("Стр. 4", 30)), H) == 4
    assert detect_page_number(ocr(("Page 5 of 9", 980)), H) == 5
    assert detect_page_number(ocr(("2/5", 980)), H) == 2


def test_numbers_in_body_ignored():
    assert detect_page_number(ocr(("Сумма 15", 500), ("3", 500)), H) is None


def test_explicit_preferred_over_bare_and_footer_over_header():
    assert detect_page_number(ocr(("2026", 20), ("Стр. 6", 970)), H) == 6
    assert detect_page_number(ocr(("4", 20), ("5", 980)), H) == 5


def test_invalid_numbers_ignored():
    assert detect_page_number(ocr(("0", 980)), H) is None
    assert detect_page_number(ocr(("7/5", 980)), H) is None
    assert detect_page_number(OcrResult(), H) is None


def test_order_by_page_numbers():
    assert order_by_page_numbers([3, 1, 2]) == [1, 2, 0]
    assert order_by_page_numbers([1, 2, 3]) is None
    # Файлы без номера остаются на месте.
    assert order_by_page_numbers([2, None, 1]) == [2, 1, 0]
    assert order_by_page_numbers([None, 5, None]) is None
    assert order_by_page_numbers([1, 1, 2]) is None
    assert has_conflicting_numbers([1, 1, None])
