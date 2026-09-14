#!/usr/bin/env python3
"""Мок-серверы для автономного тестирования: Say.action и /mission_fsm/request_pause."""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from std_srvs.srv import Trigger

from guide_robot_msgs.action import Say


class MockServicesNode(Node):
    def __init__(self) -> None:
        super().__init__("mock_services")

        # 1. Сервис паузы для mission_fsm
        self.create_service(
            Trigger,
            "/mission_fsm/request_pause",
            self._on_request_pause,
        )

        # 2. Экшен-сервер речи Say
        self._say_server = ActionServer(
            self,
            Say,
            "say",
            execute_callback=self._execute_say,
        )

        self.get_logger().info("Мок-серверы Say.action и /mission_fsm/request_pause готовы.")

    def _on_request_pause(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        self.get_logger().info("[Trigger] Получен запрос /mission_fsm/request_pause -> тур поставлен на паузу")
        response.success = True
        response.message = "paused"
        return response

    def _execute_say(self, goal_handle: object) -> Say.Result:
        text = goal_handle.request.text
        self.get_logger().info(f"[Say.action] Озвучивается реплика: \"{text}\"")
        goal_handle.succeed()
        result = Say.Result()
        result.status = Say.Result.STATUS_COMPLETED
        result.spoken_text = text
        return result


def main() -> None:
    rclpy.init()
    node = MockServicesNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
