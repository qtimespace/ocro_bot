import numpy as np
from PIL import Image

from app.models import OcrLine, OcrResult
from app.services.comments import crop_page, find_comment_cut, find_gray_pane, strip_comments
from app.services.pipeline import Pipeline
from tests.conftest import image_bytes, write

W, H = 1000, 1400


def frag(text: str, x0: float, y: float, x1: float | None = None) -> OcrLine:
    return OcrLine(text, box=(x0, y, x1 if x1 is not None else x0 + 200, y + 20))


def ocr(*fragments: OcrLine) -> OcrResult:
    return OcrResult(rows=[[f] for f in fragments])


def page_with_pane(pane_start: float | None = 0.7, color: int = 242) -> np.ndarray:
    img = np.full((H, W), 255, dtype=np.uint8)
    if pane_start is not None:
        img[100:H - 100, int(W * pane_start):] = color
    img[200:220, 100:600] = 20  # строка основного текста
    return img


BODY = [frag("ООО «Ромашка», ИНН 7701234567", 100, 200, 600), frag("Цена 125 000,00 руб.", 100, 300, 500)]


def test_strong_marker_cuts_right_column():
    comment = frag("Добавлено примечание ([ПП1]): Проверить ИНН", 720, 210)
    cut = find_comment_cut(ocr(*BODY, comment, frag("по выписке ЕГРЮЛ", 720, 235)), W)
    assert cut is not None and 600 < cut < 720
    stripped, removed = strip_comments(ocr(*BODY, comment), cut)
    assert removed == 1
    assert "Проверить" not in stripped.text and "Ромашка" in stripped.text


def test_english_word_marker():
    assert find_comment_cut(ocr(*BODY, frag("Commented [AB1]: check", 720, 210)), W) is not None


def test_weak_marker_without_pane_ignored():
    # «Примечание:» в правой колонке обычного документа — это текст документа.
    assert find_comment_cut(ocr(*BODY, frag("Примечание: оплата в рублях", 720, 210)), W) is None


def test_gray_pane_with_text_cuts():
    cut = find_comment_cut(ocr(*BODY, frag("Петров П. Заменить на 5 дней", 720, 500)), W, page_with_pane(0.7))
    assert cut is not None and abs(cut - 700) <= 5


def test_gray_pane_crossed_by_body_text_ignored():
    wide = frag("Очень длинная строка основного текста через всю страницу", 100, 500, 950)
    assert find_comment_cut(ocr(*BODY, wide), W, page_with_pane(0.7)) is None


def test_empty_gray_pane_ignored():
    assert find_comment_cut(ocr(*BODY), W, page_with_pane(0.7)) is None


def test_find_gray_pane():
    assert abs(find_gray_pane(page_with_pane(0.7)) - 0.7) < 0.01
    assert find_gray_pane(page_with_pane(None)) is None
    assert find_gray_pane(page_with_pane(0.95)) is None  # слишком узкая полоса
    assert find_gray_pane(page_with_pane(0.7, color=120)) is None  # тёмная область — не фон примечаний


def test_crop_page(tmp_path):
    path = write(tmp_path, "p.png", image_bytes("PNG", size=(600, 800)))
    assert crop_page(path, 400) == (400, 800)
    with Image.open(path) as im:
        assert im.size == (400, 800)


class CommentOcr:
    """OCR-заглушка: основной текст слева, выноска примечания справа (страница 600x800)."""

    def recognize(self, image):
        return ocr(frag("Текст договора", 50, 100, 300), frag("Добавлено примечание ([СА2]): сумма?", 420, 100, 590))


async def test_pipeline_strips_comments_and_crops(tmp_path, settings):
    work = tmp_path / "work"
    work.mkdir()
    result = await Pipeline(settings, CommentOcr(), None).run(write(work, "in", image_bytes()), work, "off")
    page = result.pages[0]
    assert page.text == "Текст договора"
    assert page.page.width < 420
    assert any("примечания Word" in w for w in result.warnings)


async def test_pipeline_comment_stripping_can_be_disabled(tmp_path):
    from tests.conftest import make_settings

    settings = make_settings(tmp_path, STRIP_WORD_COMMENTS="false")
    work = tmp_path / "work"
    work.mkdir()
    result = await Pipeline(settings, CommentOcr(), None).run(write(work, "in", image_bytes()), work, "off")
    assert "сумма?" in result.pages[0].text
    assert result.pages[0].page.width == 600


def test_side_column_of_small_text_cut():
    from app.services.comments import find_side_column

    body = [OcrLine(f"Основной текст документа, строка {i}", box=(100, 100 + 30 * i, 800, 120 + 30 * i))
            for i in range(10)]
    notes = [OcrLine(t, box=(880, 200 + 15 * i, 990, 212 + 15 * i))
             for i, t in enumerate(["Иванов И. И.", "Проверить сумму", "уточнено"])]
    assert 800 < find_side_column(ocr(*body, *notes), W) < 880
    # Такой же шрифт справа — это колонка документа, а не примечания.
    same = [OcrLine(t, box=(880, 200 + 30 * i, 990, 220 + 30 * i)) for i, t in enumerate(["Дата", "Сумма", "Итого"])]
    assert find_side_column(ocr(*body, *same), W) is None
