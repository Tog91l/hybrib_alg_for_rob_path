import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import DQN

from src.apf import normalize_force, total_force
from src.config import LOCAL_FEATURE_RADIUS, LOCAL_OBS_RADIUS, OBS_SIZE, WINDOW_SIZE
from src.map_utils import get_local_map_features, get_local_obs


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


class DqnApfAgentNode(Node):
    def __init__(self) -> None:
        super().__init__("dqn_apf_agent")

        self.declare_parameter("model_path", "/root/hydrid_alg/model/dqn_apf_hybrid_right_s.zip")
        self.declare_parameter("goal_x", 1.8)
        self.declare_parameter("goal_y", 1.8)
        self.declare_parameter("goal_tolerance", 0.20)

        self.declare_parameter("grid_size", 1000.0)
        self.declare_parameter("world_min_x", -2.2)
        self.declare_parameter("world_max_x", 2.2)
        self.declare_parameter("world_min_y", -2.2)
        self.declare_parameter("world_max_y", 2.2)

        self.declare_parameter("window_size", float(WINDOW_SIZE))
        self.declare_parameter("local_obs_radius", float(LOCAL_OBS_RADIUS))
        self.declare_parameter("local_feature_radius", float(LOCAL_FEATURE_RADIUS))
        self.declare_parameter("local_cell_meters", 0.06)
        self.declare_parameter("scan_max_range", 3.0)

        self.declare_parameter("control_hz", 5.0)
        self.declare_parameter("linear_speed", 0.14)
        self.declare_parameter("slow_linear_speed", 0.05)
        self.declare_parameter("angular_gain", 2.0)
        self.declare_parameter("max_angular_speed", 1.8)
        self.declare_parameter("max_steps", 1800)

        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")

        self.model_path = self.get_parameter("model_path").value
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

        self.grid_size = float(self.get_parameter("grid_size").value)
        self.world_min_x = float(self.get_parameter("world_min_x").value)
        self.world_max_x = float(self.get_parameter("world_max_x").value)
        self.world_min_y = float(self.get_parameter("world_min_y").value)
        self.world_max_y = float(self.get_parameter("world_max_y").value)

        self.window_size = int(self.get_parameter("window_size").value)
        self.local_obs_radius = int(self.get_parameter("local_obs_radius").value)
        self.local_feature_radius = int(self.get_parameter("local_feature_radius").value)
        self.local_cell_meters = float(self.get_parameter("local_cell_meters").value)
        self.scan_max_range = float(self.get_parameter("scan_max_range").value)

        control_hz = float(self.get_parameter("control_hz").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.slow_linear_speed = float(self.get_parameter("slow_linear_speed").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.max_steps = int(self.get_parameter("max_steps").value)

        cmd_topic = str(self.get_parameter("cmd_topic").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        scan_topic = str(self.get_parameter("scan_topic").value)

        self.model = DQN.load(self.model_path)
        self.get_logger().info(f"Loaded model: {self.model_path}")

        self.goal_grid = self.world_to_grid(self.goal_x, self.goal_y)

        self.pose: Optional[Pose2D] = None
        self.scan_msg: Optional[LaserScan] = None
        self.step_count = 0
        self.last_report_step = -100

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self.on_odom, 20)
        self.create_subscription(LaserScan, scan_topic, self.on_scan, 20)
        self.timer = self.create_timer(1.0 / control_hz, self.on_timer)

    def on_odom(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        yaw = yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.pose = Pose2D(
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw,
        )

    def on_scan(self, msg: LaserScan) -> None:
        self.scan_msg = msg

    def world_to_grid(self, x_world: float, y_world: float) -> tuple[int, int]:
        x_norm = (x_world - self.world_min_x) / max(1e-6, self.world_max_x - self.world_min_x)
        y_norm = (y_world - self.world_min_y) / max(1e-6, self.world_max_y - self.world_min_y)

        gx = int(round(clamp(x_norm, 0.0, 1.0) * (self.grid_size - 1.0)))
        gy = int(round(clamp(y_norm, 0.0, 1.0) * (self.grid_size - 1.0)))
        return gx, gy

    def make_local_map_from_scan(self, scan: LaserScan) -> np.ndarray:
        local_map = np.zeros((self.window_size, self.window_size), dtype=np.uint8)
        center = self.window_size // 2

        max_r = min(self.scan_max_range, float(scan.range_max))
        ranges = np.asarray(scan.ranges, dtype=np.float32)

        for i, r in enumerate(ranges):
            if not np.isfinite(r):
                continue
            if r <= float(scan.range_min) or r > max_r:
                continue

            angle = float(scan.angle_min) + i * float(scan.angle_increment)
            x_local = r * math.cos(angle)
            y_local = r * math.sin(angle)

            ix = int(round(center + x_local / self.local_cell_meters))
            iy = int(round(center + y_local / self.local_cell_meters))

            if 0 <= ix < self.window_size and 0 <= iy < self.window_size:
                local_map[ix, iy] = 1

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        nx, ny = ix + dx, iy + dy
                        if 0 <= nx < self.window_size and 0 <= ny < self.window_size:
                            local_map[nx, ny] = 1

        return local_map

    def build_observation(self, pose: Pose2D, scan: LaserScan) -> np.ndarray:
        local_map = self.make_local_map_from_scan(scan)
        local_obs = get_local_obs(local_map, radius=self.local_obs_radius)

        robot_grid = self.world_to_grid(pose.x, pose.y)
        half_window = local_map.shape[0] // 2
        obs_global = []

        for ox_local, oy_local in local_obs:
            gx = robot_grid[0] - half_window + ox_local
            gy = robot_grid[1] - half_window + oy_local
            obs_global.append((gx, gy))

        force = total_force(robot_grid, self.goal_grid, obs_global)
        direction = normalize_force(force)

        dx = (self.goal_grid[0] - robot_grid[0]) / self.grid_size
        dy = (self.goal_grid[1] - robot_grid[1]) / self.grid_size
        local_features = get_local_map_features(local_map, radius=self.local_feature_radius)

        obs = np.concatenate(
            [
                np.array(
                    [
                        (robot_grid[0] / self.grid_size) * 2.0 - 1.0,
                        (robot_grid[1] / self.grid_size) * 2.0 - 1.0,
                        dx,
                        dy,
                        direction[0],
                        direction[1],
                    ],
                    dtype=np.float32,
                ),
                local_features.astype(np.float32),
            ]
        ).astype(np.float32)

        if obs.shape[0] != OBS_SIZE:
            raise RuntimeError(f"Observation size mismatch: {obs.shape[0]} != {OBS_SIZE}")

        return obs

    def action_to_cmd(self, action: int, current_yaw: float) -> Twist:
        moves = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        mx, my = moves[action]
        target_yaw = math.atan2(float(my), float(mx))
        yaw_error = normalize_angle(target_yaw - current_yaw)

        angular_z = clamp(self.angular_gain * yaw_error, -self.max_angular_speed, self.max_angular_speed)

        abs_err = abs(yaw_error)
        if abs_err < 0.35:
            linear_x = self.linear_speed
        elif abs_err < 1.0:
            linear_x = self.slow_linear_speed
        else:
            linear_x = 0.0

        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        return msg

    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def on_timer(self) -> None:
        if self.pose is None or self.scan_msg is None:
            return

        dist_to_goal = math.hypot(self.goal_x - self.pose.x, self.goal_y - self.pose.y)
        if dist_to_goal <= self.goal_tolerance:
            self.stop_robot()
            self.get_logger().info(
                f"Goal reached. dist={dist_to_goal:.3f} m, steps={self.step_count}"
            )
            self.timer.cancel()
            return

        if self.step_count >= self.max_steps:
            self.stop_robot()
            self.get_logger().warn(
                f"Max steps reached ({self.max_steps}). dist={dist_to_goal:.3f} m"
            )
            self.timer.cancel()
            return

        obs = self.build_observation(self.pose, self.scan_msg)
        action, _ = self.model.predict(obs, deterministic=True)
        cmd = self.action_to_cmd(int(action), self.pose.yaw)
        self.cmd_pub.publish(cmd)

        self.step_count += 1
        if self.step_count - self.last_report_step >= 15:
            self.last_report_step = self.step_count
            self.get_logger().info(
                "step=%d action=%d pos=(%.2f, %.2f) dist=%.2f"
                % (self.step_count, int(action), self.pose.x, self.pose.y, dist_to_goal)
            )


def main() -> None:
    rclpy.init()
    node = DqnApfAgentNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
