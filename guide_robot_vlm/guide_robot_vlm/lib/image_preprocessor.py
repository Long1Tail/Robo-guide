"""Модуль: image_preprocessor.py

Подготовка кадров камеры к подаче в VLM (Vision-Language Model).
Чистая логика без rclpy: масштабирование (letterbox), сжатие в JPEG,
кодирование в Base64 Data URL и фильтрация по возрасту кадра.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

from PIL import Image


class FrameStaleError(ValueError):
    """Исключение: кадр камеры устарел (превышен max_frame_age_s)."""


class UnsupportedEncodingError(ValueError):
    """Исключение: неподдерживаемая цветовая кодировка входного кадра."""


@dataclass(frozen=True)
class PreprocessedImage:
    """Неизменяемый контейнер предобработанного кадра камеры."""

    data_url: str
    width: int
    height: int
    capture_timestamp: float
    size_bytes: int

    @property
    def raw_base64(self) -> str:
        """Чистая Base64 строка без схемы (например, для Ollama API)."""
        if "," in self.data_url:
            return self.data_url.split(",", 1)[1]
        return self.data_url

    def age_s(self, now: float | None = None) -> float:
        """Возраст кадра в секундах относительно текущего времени."""
        current = time.time() if now is None else now
        return max(0.0, current - self.capture_timestamp)


class ImagePreprocessor:
    """Препроцессор изображений для VLM-пайплайна."""

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 480,
        jpeg_quality: int = 80,
        max_frame_age_s: float = 3.0,
        letterbox: bool = True,
        letterbox_fill: tuple[int, int, int] = (128, 128, 128),
    ) -> None:
        """Инициализировать параметры предобработки."""
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"Некорректные целевые размеры: {target_width}x{target_height}")
        if not (1 <= jpeg_quality <= 100):
            raise ValueError(f"Качество JPEG должно быть в диапазоне [1..100], получено: {jpeg_quality}")

        self.target_width = target_width
        self.target_height = target_height
        self.jpeg_quality = jpeg_quality
        self.max_frame_age_s = max_frame_age_s
        self.letterbox = letterbox
        self.letterbox_fill = letterbox_fill

    def check_freshness(self, capture_timestamp: float, now: float | None = None) -> None:
        """Проверить, не устарел ли кадр. Бросает FrameStaleError при превышении max_frame_age_s."""
        if self.max_frame_age_s <= 0:
            return
        current = time.time() if now is None else now
        age = current - capture_timestamp
        if age > self.max_frame_age_s:
            raise FrameStaleError(
                f"Кадр камеры устарел: возраст {age:.3f} с > порога {self.max_frame_age_s:.3f} с"
            )

    def process_bytes(
        self,
        raw_bytes: bytes,
        width: int,
        height: int,
        encoding: str = "rgb8",
        capture_timestamp: float | None = None,
        now: float | None = None,
    ) -> PreprocessedImage:
        """Преобразовать сырой буфер пикселей из sensor_msgs/Image."""
        ts = time.time() if capture_timestamp is None else capture_timestamp
        self.check_freshness(ts, now=now)

        enc = encoding.lower()
        if enc in ("rgb8", "rgb"):
            img = Image.frombytes("RGB", (width, height), raw_bytes)
        elif enc in ("bgr8", "bgr"):
            # Pillow декодирует BGR через сырой декодер
            img = Image.frombytes("RGB", (width, height), raw_bytes, "raw", "BGR")
        elif enc in ("mono8", "grayscale", "l"):
            img = Image.frombytes("L", (width, height), raw_bytes).convert("RGB")
        elif enc in ("rgba8", "rgba"):
            img = Image.frombytes("RGBA", (width, height), raw_bytes).convert("RGB")
        elif enc in ("bgra8", "bgra"):
            img = Image.frombytes("RGB", (width, height), raw_bytes, "raw", "BGRX")
        else:
            raise UnsupportedEncodingError(f"Неподдерживаемая кодировка пикселей: {encoding!r}")

        return self.process_pil(img, capture_timestamp=ts, now=now, skip_freshness_check=True)

    def process_pil(
        self,
        image: Image.Image,
        capture_timestamp: float | None = None,
        now: float | None = None,
        skip_freshness_check: bool = False,
    ) -> PreprocessedImage:
        """Масштабировать, сжать в JPEG и закодировать PIL-изображение в Base64."""
        ts = time.time() if capture_timestamp is None else capture_timestamp
        if not skip_freshness_check:
            self.check_freshness(ts, now=now)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Ресайз с сохранением пропорций (letterbox) или обычный
        if self.letterbox:
            processed_img = self._apply_letterbox(image)
        else:
            processed_img = image.resize(
                (self.target_width, self.target_height), Image.Resampling.BILINEAR
            )

        # Сжатие в JPEG в буфер памяти
        buffer = io.BytesIO()
        processed_img.save(
            buffer,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=True,
        )
        jpeg_bytes = buffer.getvalue()
        size_bytes = len(jpeg_bytes)

        # Кодирование в Base64 Data URL
        b64_str = base64.b64encode(jpeg_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64_str}"

        return PreprocessedImage(
            data_url=data_url,
            width=processed_img.width,
            height=processed_img.height,
            capture_timestamp=ts,
            size_bytes=size_bytes,
        )

    def _apply_letterbox(self, image: Image.Image) -> Image.Image:
        """Пропорциональное масштабирование с добавлением полей (letterbox padding)."""
        src_w, src_h = image.size
        dst_w, dst_h = self.target_width, self.target_height

        scale = min(dst_w / src_w, dst_h / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))

        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        canvas = Image.new("RGB", (dst_w, dst_h), self.letterbox_fill)
        pad_x = (dst_w - new_w) // 2
        pad_y = (dst_h - new_h) // 2
        canvas.paste(resized, (pad_x, pad_y))
        return canvas


# Синоним (alias) для удобства импорта под именем ImageProcessor
ImageProcessor = ImagePreprocessor

__all__ = [
    "FrameStaleError",
    "ImagePreprocessor",
    "ImageProcessor",
    "PreprocessedImage",
    "UnsupportedEncodingError",
]
