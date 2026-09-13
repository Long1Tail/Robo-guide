"""Юнит-тесты для модуля image_preprocessor.py (чистая логика без ROS)."""

from __future__ import annotations

import base64
import io
import time

import pytest
from PIL import Image

from guide_robot_vlm.lib.image_preprocessor import (
    FrameStaleError,
    ImagePreprocessor,
    ImageProcessor,
    PreprocessedImage,
    UnsupportedEncodingError,
)


def test_alias() -> None:
    """Проверка, что ImageProcessor является алиасом ImagePreprocessor."""
    assert ImageProcessor is ImagePreprocessor


def test_invalid_parameters() -> None:
    """Проверка валидации входных аргументов конструктора."""
    with pytest.raises(ValueError, match="Некорректные целевые размеры"):
        ImagePreprocessor(target_width=0, target_height=480)

    with pytest.raises(ValueError, match="Качество JPEG должно быть"):
        ImagePreprocessor(jpeg_quality=105)


def test_frame_stale_check() -> None:
    """Проверка отбрасывания устаревших кадров по таймстампу."""
    prep = ImagePreprocessor(max_frame_age_s=1.0)
    now = 100.0

    # Свежий кадр (возраст 0.2 сек) -- проходит без ошибок
    prep.check_freshness(capture_timestamp=99.8, now=now)

    # Устаревший кадр (возраст 1.5 сек) -- FrameStaleError
    with pytest.raises(FrameStaleError, match="Кадр камеры устарел"):
        prep.check_freshness(capture_timestamp=98.5, now=now)


def test_preprocessed_image_properties() -> None:
    """Проверка свойств PreprocessedImage (data_url, raw_base64, age_s)."""
    raw_b64 = "iVBORw0KGgoAAAANSUhEUgAA"
    data_url = f"data:image/jpeg;base64,{raw_b64}"
    img = PreprocessedImage(
        data_url=data_url,
        width=640,
        height=480,
        capture_timestamp=10.0,
        size_bytes=100,
    )

    assert img.raw_base64 == raw_b64
    assert img.age_s(now=12.5) == 2.5
    assert img.width == 640
    assert img.height == 480


def test_process_pil_letterbox() -> None:
    """Проверка пропорционального масштабирования (letterbox)."""
    prep = ImagePreprocessor(target_width=640, target_height=480, jpeg_quality=80, letterbox=True)

    # Исходное изображение 1000x500 (2:1 соотношение)
    src_img = Image.new("RGB", (1000, 500), color=(255, 0, 0))
    res = prep.process_pil(src_img, capture_timestamp=time.time())

    assert res.width == 640
    assert res.height == 480
    assert res.data_url.startswith("data:image/jpeg;base64,")
    assert res.size_bytes > 0

    # Проверяем, что Base64 декодируется обратно в валидный JPEG
    b64_data = res.raw_base64
    decoded_bytes = base64.b64decode(b64_data)
    decoded_img = Image.open(io.BytesIO(decoded_bytes))
    assert decoded_img.size == (640, 480)
    assert decoded_img.format == "JPEG"


def test_process_bytes_rgb8() -> None:
    """Проверка обработки сырого буфера RGB8 (как из sensor_msgs/Image)."""
    prep = ImagePreprocessor(target_width=320, target_height=240)
    w, h = 100, 100
    # Создаем 100x100 зеленый квадрат: R=0, G=255, B=0
    raw_rgb = bytes([0, 255, 0] * (w * h))

    res = prep.process_bytes(
        raw_bytes=raw_rgb,
        width=w,
        height=h,
        encoding="rgb8",
        capture_timestamp=time.time(),
    )

    assert res.width == 320
    assert res.height == 240
    assert len(res.raw_base64) > 0


def test_process_bytes_bgr8() -> None:
    """Проверка обработки BGR8 (стандарт OpenCV/ROS)."""
    prep = ImagePreprocessor(target_width=320, target_height=240)
    w, h = 50, 50
    # Синий цвет в BGR: B=255, G=0, R=0 -> в RGB должен стать R=0, G=0, B=255
    raw_bgr = bytes([255, 0, 0] * (w * h))

    res = prep.process_bytes(
        raw_bytes=raw_bgr,
        width=w,
        height=h,
        encoding="bgr8",
        capture_timestamp=time.time(),
    )

    decoded = Image.open(io.BytesIO(base64.b64decode(res.raw_base64)))
    assert decoded.size == (320, 240)


def test_unsupported_encoding() -> None:
    """Проверка исключения на неизвестную кодировку."""
    prep = ImagePreprocessor()
    with pytest.raises(UnsupportedEncodingError, match="Неподдерживаемая кодировка"):
        prep.process_bytes(b"\x00\x00", 1, 1, encoding="unknown_codec_123")
