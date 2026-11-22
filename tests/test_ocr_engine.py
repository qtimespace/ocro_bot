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


def test_paddle_v3_called_with_fast_defaults(monkeypatch):
    import sys
    import types

    captured = {}

    class FakePaddle:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, image):
            return []

    monkeypatch.setitem(sys.modules, "paddleocr", types.SimpleNamespace(PaddleOCR=FakePaddle))
    PaddleOcrEngine(["ru"]).recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert captured["use_doc_orientation_classify"] is False
    assert captured["use_textline_orientation"] is False
    assert captured["text_det_limit_type"] == "max" and captured["text_det_limit_side_len"] == 1280

    captured.clear()
    PaddleOcrEngine(["ru"], orientation=True, det_limit_side=0).recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert captured["use_textline_orientation"] is True
    assert "text_det_limit_type" not in captured


def test_parse_rapid_output():
    from types import SimpleNamespace

    from app.services.ocr_engine import parse_rapid_output

    result = SimpleNamespace(
        boxes=np.array([[[10, 50], [90, 50], [90, 70], [10, 70]], [[10, 10], [90, 10], [90, 30], [10, 30]]]),
        txts=("второй", "first"), scores=(0.9, 0.5),
    )
    fragments = parse_rapid_output(result)
    assert [(f.text, f.confidence) for f in fragments] == [("второй", 0.9), ("first", 0.5)]
    assert fragments[1].box == (10, 10, 90, 30)
    assert parse_rapid_output(SimpleNamespace(boxes=None, txts=None, scores=None)) == []


def test_engine_factory(tmp_path):
    from app.services.ocr_engine import RapidOcrEngine, create_ocr_engine
    from tests.conftest import make_settings

    onnx = create_ocr_engine(make_settings(tmp_path))
    assert isinstance(onnx, RapidOcrEngine) and onnx.rec_lang == "eslav" and onnx.det_limit_side == 1280
    assert create_ocr_engine(make_settings(tmp_path, OCR_REC_LANG="cyrillic")).rec_lang == "cyrillic"
    assert isinstance(create_ocr_engine(make_settings(tmp_path, OCR_BACKEND="paddle")), PaddleOcrEngine)


def test_rapid_engine_wraps_errors():
    from app.services.ocr_engine import RapidOcrEngine

    class Broken:
        def __call__(self, image):
            raise RuntimeError("boom")

    engine = RapidOcrEngine(["ru"])
    engine._engine = Broken()
    with pytest.raises(OcrError):
        engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
