"""Модуль: tool_broker.py

ОПИСАНИЕ БУДУЩЕГО ФУНКЦИОНАЛА
=============================

Назначение:
-----------
Исполнитель навыков (Tool Broker). Отвечает за трансляцию проверенных решений
модели (ValidatedAction) в вызовы стандартных ROS 2 интерфейсов (Actions/Services).

Маппинг навыков на ROS 2 Action интерфейсы:
------------------------------------------
1. `start_tour(tour_id)`:
   - Отправляет цель в экшен `run_tour` (`guide_robot_msgs/action/RunTour.action`).
   - Инициирует движение робота по маршруту экскурсии через mission_fsm.

2. `stop_tour()`:
   - Вызывает отмену активной цели `RunTour.action`.

3. `goto_exhibit(location_id)`:
   - Отправляет команду навигации к указанной точке экспозиции.

4. `describe_exhibit(content_id)`:
   - Отправляет цель в экшен `narrate` (`guide_robot_msgs/action/Narrate.action`).
   - Запускает озвучивание кураторского текста экспоната через narration_server.

5. `ask_visitor(question_id)`:
   - Отправляет цель в экшен `ask_user` (`guide_robot_msgs/action/AskUser.action`).
   - Задает вопрос посетителям и ожидает ответа.

6. `wait()` / `idle()`:
   - Удержание текущей паузы или холостой режим без отправки новых команд.

7. Озвучивание безопасных фолбэков:
   - Метод `speak_fallback(text)`: отправка фразы уточнения («Повторите, пожалуйста...»)
     в `say` (`guide_robot_msgs/action/Say.action`) с приоритетом диалога (priority=40).
"""
