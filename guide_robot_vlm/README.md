# `guide_robot_vlm`

Пакет мультимодального оркестратора принятия решений робота-экскурсовода на базе **Vision-Language Model (VLM)**.

## 1. Назначение и рамки

Модуль заменяет чисто текстовый контур диалога (`guide_robot_llm`) на мультимодальный узел принятия решений, работающий по связке:
$$\text{(Кадр с камеры + Запрос посетителя + Состояние миссии)} \longrightarrow \text{Выбор навыка (Tool Call в JSON)} \longrightarrow \text{Вызов в ToolBroker}$$

### Входит в состав пакета:
- `vlm_agent_node`: ROS 2 `LifecycleNode`, синхронизирующий входящие потоки с камеры, ASR и FSM миссии.
- `lib/image_preprocessor.py`: захват и оптимизация кадров камеры (масштабирование, JPEG-компрессия, Base64).
- `lib/vlm_backends.py`: клиент инференса VLM (vLLM, llama.cpp, mock) с поддержкой принудительной схемы (JSON Schema).
- `lib/validator.py`: двухуровневая защита и семантическая валидация аргументов (`location_id`, `content_id`) по данным `guide_robot_semantic_map`.
- `lib/tool_broker.py`: адаптер вызова навыков робота через ROS 2 Actions (`RunTour`, `Narrate`, `AskUser`, `Say`).
- `lib/history.py` и `lib/turn_log.py`: кольцевой буфер контекста и сессионный JSONL-лог ходов.

### Не входит в задачу:
- Модель **не** генерирует произвольные описания экспонатов (все тексты берутся по `content_id` из `guide_robot_semantic_map`).
- Модель **не** вычисляет траектории колес (навигация делегируется Nav2).
- Модель **не** затрагивает контур безопасности (`collision_monitor`, E-Stop).

## 2. Библиотека навыков (Tools)

- `start_tour(tour_id)`: старт тура через `RunTour.action`.
- `stop_tour()`: остановка текущего тура.
- `goto_exhibit(location_id)`: перемещение к точке.
- `describe_exhibit(content_id)`: запуск рассказа об экспонате через `Narrate.action`.
- `ask_visitor(question_id)`: вопрос посетителю через `AskUser.action`.
- `wait()` / `idle()`: пауза / безопасный режим ожидания.

## 3. Запуск

```bash
# Запуск с автостартом lifecycle-ноды
ros2 launch guide_robot_vlm vlm.launch.py autostart:=true
```
