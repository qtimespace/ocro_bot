import numpy as np
import pytest

from app.models import OcrLine
from app.services.ocr_engine import OcrError, PaddleOcrEngine, order_fragments, paddle_language, parse_paddle_output


def test_language_mapping():
    assert paddle_language(["ru", "en"]) == "ru"
    assert paddle_language(["en"]) == "en"
    assert paddle_language([]) == "en"


def test_reading_order_rows_and_columns():
    fragments = [
        OcrLine("right", box=(300, 102, 400, 122)),
        OcrLine("second", box=(10, 150, 100, 170)),
        OcrLine("left", box=(10, 100, 100, 120)),
        OcrLine("title", box=(10, 10, 200, 30)),
        OcrLine("  ", box=(10, 200, 20, 210)),
    ]
    rows = order_fragments(fragments)
    assert [[f.text for f in row] for row in rows] == [["title"], ["left", "right"], ["second"]]


def test_parse_paddle_v2_output():
    raw = [[
        [[[10, 50], [100, 50], [100, 70], [10, 70]], ("второй", 0.9)],
        [[[10, 10], [100, 10], [100, 30], [10, 30]], ("first", 0.8)],
    ]]
    fragments = parse_paddle_output(raw)
    assert [f.text for f in fragments] == ["второй", "first"]
    assert fragments[1].box == (10, 10, 100, 30)
    assert [f.text for row in order_fragments(fragments) for f in row] == ["first", "второй"]


def test_parse_paddle_v2_empty_page():
    assert parse_paddle_output([None]) == []
    assert parse_paddle_output(None) == []


def test_parse_paddle_v3_output():
    raw = [{
        "rec_texts": ["Hello", "Мир"],
        "rec_scores": np.array([0.99, 0.5]),
        "rec_boxes": np.array([[10, 10, 50, 30], [60, 10, 90, 30]]),
    }]
    fragments = parse_paddle_output(raw)
    assert [(f.text, round(f.confidence, 2)) for f in fragments] == [("Hello", 0.99), ("Мир", 0.5)]
    assert fragments[1].box == (60, 10, 90, 30)


def test_parse_paddle_v3_polys_fallback():
    raw = [{"rec_texts": ["A"], "rec_scores": [0.9], "rec_polys": [np.array([[0, 0], [5, 0], [5, 5], [0, 5]])]}]
    assert parse_paddle_output(raw)[0].box == (0, 0, 5, 5)


def test_engine_wraps_backend_errors():
    class Broken:
        def predict(self, image):
            raise RuntimeError("boom")

    engine = PaddleOcrEngine(["ru"])
    engine._engine, engine._api_v3 = Broken(), True
    with pytest.raises(OcrError):
        engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))


def test_engine_v3_end_to_end_with_stub():
    class Stub:
        def predict(self, image):
            return [{"rec_texts": ["B", "A"], "rec_scores": [0.9, 0.9],
                     "rec_boxes": [[0, 50, 10, 60], [0, 0, 10, 10]]}]

    engine = PaddleOcrEngine(["ru"])
    engine._engine, engine._api_v3 = Stub(), True
    assert engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8)).text == "A\nB"


def test_rec_model_follows_language_when_det_model_set():
    assert PaddleOcrEngine(["ru", "en"]).rec_model == "eslav_PP-OCRv5_mobile_rec"
    assert PaddleOcrEngine(["en"]).rec_model == "en_PP-OCRv5_mobile_rec"
    assert PaddleOcrEngine(["ru"], rec_model="custom").rec_model == "custom"
    assert PaddleOcrEngine(["ru"], det_model="").rec_model == ""
    with pytest.raises(ValueError):
        PaddleOcrEngine(["de"])
