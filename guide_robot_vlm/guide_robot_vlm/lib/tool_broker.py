"""Исполнение VLM-tools через ROS 2 интерфейсы."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rclpy.node import Node
from rclpy.task import Future
from std_srvs.srv import Trigger

if TYPE_CHECKING:
    from guide_robot_vlm.lib.validator import ValidatedAction


class ToolBroker:
    """
    Исполнитель проверенных действий VLM.

    Сейчас поддерживаются:
    - idle: ничего не менять;
    - interrupt: поставить активный тур на паузу.
    """

    def __init__(self, node: Node) -> None:
        self._node = node

        self._pause_client = node.create_client(
            Trigger,
            "/mission_fsm/request_pause",
        )

        self._pause_request_in_flight = False

    def dispatch(self, action: "ValidatedAction") -> bool:
        """Выполнить уже провалидированное действие."""
        tool = action.tool.strip().lower()

        if tool == "idle":
            return self.idle(action)

        if tool == "interrupt":
            return self.interrupt(action)

        self._node.get_logger().warning(
            f"ToolBroker: неподдерживаемый tool={action.tool!r}"
        )
        return False

    def idle(self, action: "ValidatedAction") -> bool:
        """Ничего не делать."""
        if action.is_fallback:
            self._node.get_logger().debug(
                "ToolBroker: idle fallback: "
                f"{action.fallback_reason!r}"
            )

        return True

    def interrupt(self, action: "ValidatedAction") -> bool:
        """
        Прервать текущий рассказ и поставить тур на паузу.

        Автоматическое возобновление здесь не реализуется.
        """
        self._node.get_logger().info(
            "ToolBroker: interrupt: "
            f"confidence={action.confidence:.3f}, "
            f"args={action.args!r}"
        )

        if self._pause_request_in_flight:
            self._node.get_logger().debug(
                "ToolBroker: request_pause уже выполняется"
            )
            return True

        if not self._pause_client.service_is_ready():
            self._node.get_logger().warning(
                "ToolBroker: сервис "
                "/mission_fsm/request_pause недоступен"
            )
            return False

        self._pause_request_in_flight = True

        try:
            future = self._pause_client.call_async(
                Trigger.Request()
            )
        except Exception as error:
            self._pause_request_in_flight = False
            self._node.get_logger().error(
                f"ToolBroker: не удалось отправить request_pause: {error}"
            )
            return False

        future.add_done_callback(
            self._on_pause_response
        )

        self._node.get_logger().info(
            "ToolBroker: отправлен request_pause"
        )

        return True

    def _on_pause_response(self, future: Future) -> None:
        """Обработать ответ mission_fsm на request_pause."""
        self._pause_request_in_flight = False

        try:
            response = future.result()
        except Exception as error:
            self._node.get_logger().error(
                f"ToolBroker: ошибка request_pause: {error}"
            )
            return

        if response is None:
            self._node.get_logger().error(
                "ToolBroker: request_pause вернул пустой ответ"
            )
            return

        if not response.success:
            detail = response.message or "неизвестная причина"
            self._node.get_logger().warning(
                f"ToolBroker: пауза не выполнена: {detail}"
            )
            return

        self._node.get_logger().info(
            "ToolBroker: тур поставлен на паузу"
        )