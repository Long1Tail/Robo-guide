"""Юнит-тесты для модуля validator.py (Этап 1: idle и interrupt)."""

from __future__ import annotations

import pytest

from guide_robot_vlm.lib.validator import (
    ActionValidator,
    SemanticContext,
    ValidatedAction,
    extract_and_parse_json,
)


def test_extract_and_parse_json_variants() -> None:
    """Проверка извлечения JSON из чистого текста, markdown и текста с преамбулой."""
    # 1. Чистый JSON
    raw = '{"tool": "interrupt", "args": {}, "confidence": 0.9, "abstain": false}'
    res, err = extract_and_parse_json(raw)
    assert err is None
    assert res == {"tool": "interrupt", "args": {}, "confidence": 0.9, "abstain": False}

    # 2. Markdown код-блок ```json ... ```
    md_raw = 'Here is the decision:\n```json\n{"tool": "idle", "args": {}, "confidence": 0.8, "abstain": false}\n```'
    res, err = extract_and_parse_json(md_raw)
    assert err is None
    assert res["tool"] == "idle"

    # 3. Текст с преамбулой без markdown
    preamble = 'Thinking: The user wants to interrupt. {"tool": "interrupt", "args": {"reason": "no_audience"}, "confidence": 0.95, "abstain": false} Thank you.'
    res, err = extract_and_parse_json(preamble)
    assert err is None
    assert res["tool"] == "interrupt"

    # 4. Сломанный JSON
    broken = '{"tool": "interrup'
    res, err = extract_and_parse_json(broken)
    assert res is None
    assert "Ошибка синтаксиса" in err


def test_validator_valid_interrupt_action() -> None:
    """Проверка успешной валидации корректного действия interrupt (оценка аудитории)."""
    validator = ActionValidator(min_confidence=0.7)

    raw_json = """
    {
        "tool": "interrupt",
        "args": {
            "reason": "no_audience",
            "people_count": 0,
            "looking_at_robot": false
        },
        "confidence": 0.88,
        "abstain": false
    }
    """
    action = validator.validate(raw_json)
    assert action.is_fallback is False
    assert action.tool == "interrupt"
    assert action.args["reason"] == "no_audience"
    assert action.args["people_count"] == 0
    assert action.args["looking_at_robot"] is False
    assert action.confidence == 0.88
    assert action.fallback_reason is None


def test_validator_interrupt_invalid_args() -> None:
    """Проверка отклонения некорректных типов аргументов для interrupt."""
    validator = ActionValidator()

    # Некорректный тип people_count (строка вместо int)
    raw_bad_count = '{"tool": "interrupt", "args": {"people_count": "none"}, "confidence": 0.9, "abstain": false}'
    action1 = validator.validate(raw_bad_count)
    assert action1.is_fallback is True
    assert "invalid_arg" in action1.fallback_reason

    # Некорректный тип looking_at_robot (число вместо bool)
    raw_bad_looking = '{"tool": "interrupt", "args": {"looking_at_robot": 123}, "confidence": 0.9, "abstain": false}'
    action2 = validator.validate(raw_bad_looking)
    assert action2.is_fallback is True
    assert "invalid_arg" in action2.fallback_reason


def test_validator_low_confidence_fallback() -> None:
    """Проверка срабатывания отката при низкой уверенности."""
    validator = ActionValidator(min_confidence=0.7)
    raw = '{"tool": "interrupt", "args": {}, "confidence": 0.65, "abstain": false}'

    action = validator.validate(raw)
    assert action.is_fallback is True
    assert action.tool == "idle"
    assert "low_confidence" in action.fallback_reason


def test_validator_abstain_fallback() -> None:
    """Проверка срабатывания отката при флаге abstain: true."""
    validator = ActionValidator()
    raw = '{"tool": "interrupt", "args": {}, "confidence": 0.9, "abstain": true}'

    action = validator.validate(raw)
    assert action.is_fallback is True
    assert action.tool == "idle"
    assert action.fallback_reason == "model_abstained"


def test_validator_disabled_tools_rejected_as_unknown() -> None:
    """Проверка, что отключенные на Этапе 1 инструменты отклоняются как unknown_tool."""
    validator = ActionValidator()

    for disabled_tool in ("goto_exhibit", "start_tour", "describe_exhibit", "wait", "chat"):
        raw = f'{{"tool": "{disabled_tool}", "args": {{}}, "confidence": 0.99, "abstain": false}}'
        action = validator.validate(raw)
        assert action.is_fallback is True
        assert "unknown_tool" in action.fallback_reason


def test_validator_missing_fields_fallback() -> None:
    """Проверка отката при отсутствии обязательных полей."""
    validator = ActionValidator()
    # Нет поля abstain
    raw = '{"tool": "interrupt", "args": {}, "confidence": 0.95}'
    action = validator.validate(raw)

    assert action.is_fallback is True
    assert "missing_fields: abstain" in action.fallback_reason


def test_validator_broken_json_fallback() -> None:
    """Проверка отката при абсолютно невалидном тексте модели."""
    validator = ActionValidator()
    action = validator.validate("I am a robot and I don't know what to do!")

    assert action.is_fallback is True
    assert action.tool == "idle"
    assert "invalid_json" in action.fallback_reason
