"""Модуль: validator.py

Семантическая и синтаксическая валидация вывода VLM (Guardrails).
Чистая логика без rclpy:
1. Безопасный парсинг JSON с удалением markdown-оберток и отловом синтаксических ошибок.
2. Проверка наличия и типов обязательных полей (tool, args, confidence, abstain).
3. Проверка допустимости имени навыка (закрытый список ALLOWED_TOOLS).
4. Проверка уверенности (confidence >= min_confidence) и флага abstain.
5. Семантическая валидация аргументов (location_id, content_id, tour_id) по семантической карте.
6. Формирование гарантированно безопасного действия ValidatedAction (с fallback при любых сбоях).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Закрытый список разрешенных навыков согласно ТЗ (разд. 5) + диалоговый chat
DEFAULT_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {
        # "start_tour",
        # "stop_tour",
        # "goto_exhibit",
        # "describe_exhibit",
        # "ask_visitor",
        # "wait",
        "idle",
        #"chat",
        "interrupt"
    }
)


@dataclass(frozen=True)
class SemanticContext:
    """Белые списки идентификаторов из guide_robot_semantic_map для валидации."""

    allowed_locations: frozenset[str] = field(default_factory=frozenset)
    allowed_contents: frozenset[str] = field(default_factory=frozenset)
    allowed_tours: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ValidatedAction:
    """Гарантированно безопасное и типизированное действие для передачи в tool_broker."""

    tool: str
    args: dict[str, Any]
    confidence: float
    is_fallback: bool
    fallback_reason: str | None = None
    fallback_phrase: str = ""
    raw_json: dict[str, Any] | None = None


def extract_and_parse_json(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Безопасно извлечь и распарсить JSON из строки ответа модели.

    Обрабатывает:
    - Сырой JSON: `{"tool": ...}`
    - Markdown code blocks: ```json {"tool": ...} ```
    - Вводные фразы до/после JSON.
    Возвращает: (parsed_dict, None) в случае успеха или (None, error_description).
    """
    if not raw_text or not raw_text.strip():
        return None, "Пустой ответ модели"

    text = raw_text.strip()

    # 1. Снимаем markdown code block ```json ... ``` если присутствует
    if "```" in text:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # 2. Если вокруг JSON есть текст, находим внешние фигурные скобки { ... }
    if not (text.startswith("{") and text.endswith("}")):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]

    # 3. Парсинг JSON с отловом синтаксических ошибок
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return None, f"JSON корень должен быть объектом (dict), получен {type(data).__name__}"
        return data, None
    except json.JSONDecodeError as err:
        return None, f"Ошибка синтаксиса JSON: {err}"


class ActionValidator:
    """Валидатор решений VLM-модели с многоуровневой защитой."""

    def __init__(
        self,
        min_confidence: float = 0.70,
        fallback_phrase: str = "Извините, я не совсем понял вас. Пожалуйста, повторите.",
        allowed_tools: frozenset[str] | None = None,
    ) -> None:
        """Инициализировать валидатор с заданными порогами и фразой отката."""
        self.min_confidence = float(min_confidence)
        self.fallback_phrase = fallback_phrase
        self.allowed_tools = allowed_tools if allowed_tools is not None else DEFAULT_ALLOWED_TOOLS

    def create_fallback(self, reason: str, raw_json: dict[str, Any] | None = None) -> ValidatedAction:
        """Сформировать безопасное действие отката (idle)."""
        return ValidatedAction(
            tool="idle",
            args={},
            confidence=0.0,
            is_fallback=True,
            fallback_reason=reason,
            fallback_phrase=self.fallback_phrase,
            raw_json=raw_json,
        )

    def validate(
        self,
        raw_output: str | dict[str, Any],
        context: SemanticContext | None = None,
    ) -> ValidatedAction:
        """Выполнить полную трехуровневую валидацию вывода VLM.

        1. Уровень 1: Синтаксис JSON.
        2. Уровень 2: Структура, типы полей, confidence, abstain, имя навыка.
        3. Уровень 3: Семантическая проверка аргументов по SemanticContext.
        """
        # --- Уровень 1: Синтаксический разбор ---
        if isinstance(raw_output, str):
            data, parse_err = extract_and_parse_json(raw_output)
            if data is None:
                return self.create_fallback(f"invalid_json: {parse_err}")
        elif isinstance(raw_output, dict):
            data = raw_output
        else:
            return self.create_fallback(f"invalid_input_type: {type(raw_output).__name__}")

        # --- Уровень 2: Структурная проверка полей и типов ---
        # 1. Проверка наличия обязательных полей верхнего уровня
        required_fields = ("tool", "args", "confidence", "abstain")
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return self.create_fallback(
                f"missing_fields: {', '.join(missing_fields)}", raw_json=data
            )

        tool = data["tool"]
        args = data["args"]
        confidence_val = data["confidence"]
        abstain = data["abstain"]

        if not isinstance(tool, str):
            return self.create_fallback(
                f"invalid_field_type: 'tool' must be str, got {type(tool).__name__}", raw_json=data
            )
        if not isinstance(args, dict):
            return self.create_fallback(
                f"invalid_field_type: 'args' must be dict, got {type(args).__name__}", raw_json=data
            )
        if not isinstance(abstain, bool):
            return self.create_fallback(
                f"invalid_field_type: 'abstain' must be bool, got {type(abstain).__name__}",
                raw_json=data,
            )

        try:
            confidence = float(confidence_val)
        except (ValueError, TypeError):
            return self.create_fallback(
                f"invalid_field_type: 'confidence' must be float, got {confidence_val!r}",
                raw_json=data,
            )

        # 2. Проверка флага осознанного отказа (abstain)
        if abstain is True:
            return self.create_fallback("model_abstained", raw_json=data)

        # 3. Проверка порога уверенности модели
        if confidence < self.min_confidence:
            return self.create_fallback(
                f"low_confidence: {confidence:.2f} < {self.min_confidence:.2f}",
                raw_json=data,
            )

        # 4. Проверка имени навыка по закрытому списку
        tool_clean = tool.strip()
        if tool_clean not in self.allowed_tools:
            return self.create_fallback(f"unknown_tool: {tool_clean!r}", raw_json=data)

        # --- Уровень 3: Семантическая валидация аргументов ---
        semantic_err = self._validate_tool_args(tool_clean, args, context)
        if semantic_err is not None:
            return self.create_fallback(semantic_err, raw_json=data)

        # Все проверки успешно пройдены!
        return ValidatedAction(
            tool=tool_clean,
            args=args,
            confidence=confidence,
            is_fallback=False,
            raw_json=data,
        )

    def _validate_tool_args(
        self,
        tool: str,
        args: dict[str, Any],
        context: SemanticContext | None,
    ) -> str | None:
        """Проверить аргументы навыка на корректность."""
        del context  # Не используется на Этапе 1

        if tool == "interrupt":
            # Валидация аргументов для навыка прерывания (оценка аудитории / стоп)
            reason = args.get("reason")
            if reason is not None and not isinstance(reason, str):
                return "invalid_arg: interrupt 'reason' must be string"

            people_count = args.get("people_count")
            if people_count is not None:
                if not isinstance(people_count, int) or isinstance(people_count, bool) or people_count < 0:
                    return "invalid_arg: interrupt 'people_count' must be non-negative integer"

            looking_at_robot = args.get("looking_at_robot")
            if looking_at_robot is not None and not isinstance(looking_at_robot, bool):
                return "invalid_arg: interrupt 'looking_at_robot' must be boolean"

            return None

        # --- Инструменты временно отключены для Этапа 1 (будут включены на следующих этапах) ---
        # elif tool == "goto_exhibit":
        #     location_id = args.get("location_id")
        #     if not location_id or not isinstance(location_id, str):
        #         return "missing_arg: goto_exhibit requires non-empty string 'location_id'"
        #     if context and context.allowed_locations and location_id not in context.allowed_locations:
        #         return f"unknown_location: {location_id!r} not in semantic map locations"
        #
        # elif tool == "describe_exhibit":
        #     content_id = args.get("content_id")
        #     if not content_id or not isinstance(content_id, str):
        #         return "missing_arg: describe_exhibit requires non-empty string 'content_id'"
        #     if context and context.allowed_contents and content_id not in context.allowed_contents:
        #         return f"unknown_content: {content_id!r} not in semantic map exhibits"
        #
        # elif tool == "start_tour":
        #     tour_id = args.get("tour_id")
        #     if tour_id is not None:
        #         if not isinstance(tour_id, str):
        #             return "invalid_arg: start_tour 'tour_id' must be string"
        #         if context and context.allowed_tours and tour_id not in context.allowed_tours:
        #             return f"unknown_tour: {tour_id!r} not in semantic map tours"

        return None


__all__ = [
    "DEFAULT_ALLOWED_TOOLS",
    "ActionValidator",
    "SemanticContext",
    "ValidatedAction",
    "extract_and_parse_json",
]
