import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import MapMetaData
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import DQN
from tf2_ros import Buffer, TransformException, TransformListener

from src.apf import normalize_force, total_force
from src.config import LOCAL_FEATURE_RADIUS, LOCAL_OBS_RADIUS, OBS_SIZE, WINDOW_SIZE
from src.map_utils import get_local_map_features, get_local_obs


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


class HybridNav2Bridge(Node):
    def __init__(self) -> None:
        super().__init__("hybrid_nav2_bridge")

        self.declare_parameter("model_path", "/root/hydrid_alg/model/dqn_apf_hybrid_right_s.zip")
        self.declare_parameter("use_goal_topic", True)
        self.declare_parameter("goal_topic", "/goal_pose")
        self.declare_parameter("final_goal_x", float("nan"))
        self.declare_parameter("final_goal_y", float("nan"))
        self.declare_parameter("final_goal_tolerance", 0.25)
        self.declare_parameter("direct_goal_radius", 0.8)

        self.declare_parameter("grid_size", 1000.0)
        self.declare_parameter("world_min_x", -5.0)
        self.declare_parameter("world_max_x", 5.0)
        self.declare_parameter("world_min_y", -5.0)
        self.declare_parameter("world_max_y", 5.0)

        self.declare_parameter("window_size", float(WINDOW_SIZE))
        # Keep compatibility with the original training code, where obstacle projection
        # used a fixed half-window of 30 (from legacy LOCAL_SIZE=60), not window_size/2.
        self.declare_parameter("obs_projection_half_window", 30.0)
        self.declare_parameter("local_obs_radius", float(LOCAL_OBS_RADIUS))
        self.declare_parameter("local_feature_radius", float(LOCAL_FEATURE_RADIUS))
        self.declare_parameter("local_cell_meters", 0.08)
        self.declare_parameter("scan_max_range", 3.0)

        self.declare_parameter("control_hz", 1.5)
        self.declare_parameter("subgoal_step_m", 0.35)
        self.declare_parameter("subgoal_tolerance_m", 0.20)
        self.declare_parameter("subgoal_timeout_sec", 6.0)
        self.declare_parameter("action_swap_xy", False)
        self.declare_parameter("away_from_goal_margin_m", 0.05)
        self.declare_parameter("recovery_timeout_replans", 3)
        self.declare_parameter("recovery_step_scale", 2.0)
        self.declare_parameter("stagnation_dist_epsilon_m", 0.03)
        self.declare_parameter("stagnation_decisions_threshold", 6)
        self.declare_parameter("escape_clearance_weight", 0.35)
        self.declare_parameter("escape_goal_weight", 1.0)
        self.declare_parameter("escape_min_clearance_m", 0.25)
        self.declare_parameter("bounds_padding_m", 0.10)
        self.declare_parameter("use_map_metadata_bounds", True)
        self.declare_parameter("map_metadata_topic", "/map_metadata")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("nav_action_name", "navigate_to_pose")

        self.model_path = str(self.get_parameter("model_path").value)
        self.use_goal_topic = bool(self.get_parameter("use_goal_topic").value)
        goal_topic = str(self.get_parameter("goal_topic").value)

        self.final_goal_tolerance = float(self.get_parameter("final_goal_tolerance").value)
        self.direct_goal_radius = float(self.get_parameter("direct_goal_radius").value)

        self.grid_size = float(self.get_parameter("grid_size").value)
        self.world_min_x = float(self.get_parameter("world_min_x").value)
        self.world_max_x = float(self.get_parameter("world_max_x").value)
        self.world_min_y = float(self.get_parameter("world_min_y").value)
        self.world_max_y = float(self.get_parameter("world_max_y").value)

        self.window_size = int(self.get_parameter("window_size").value)
        self.obs_projection_half_window = int(self.get_parameter("obs_projection_half_window").value)
        self.local_obs_radius = int(self.get_parameter("local_obs_radius").value)
        self.local_feature_radius = int(self.get_parameter("local_feature_radius").value)
        self.local_cell_meters = float(self.get_parameter("local_cell_meters").value)
        self.scan_max_range = float(self.get_parameter("scan_max_range").value)

        control_hz = float(self.get_parameter("control_hz").value)
        self.subgoal_step_m = float(self.get_parameter("subgoal_step_m").value)
        self.subgoal_tolerance_m = float(self.get_parameter("subgoal_tolerance_m").value)
        self.subgoal_timeout_sec = float(self.get_parameter("subgoal_timeout_sec").value)
        self.action_swap_xy = bool(self.get_parameter("action_swap_xy").value)
        self.away_from_goal_margin_m = float(self.get_parameter("away_from_goal_margin_m").value)
        self.recovery_timeout_replans = int(self.get_parameter("recovery_timeout_replans").value)
        self.recovery_step_scale = float(self.get_parameter("recovery_step_scale").value)
        self.stagnation_dist_epsilon_m = float(self.get_parameter("stagnation_dist_epsilon_m").value)
        self.stagnation_decisions_threshold = int(
            self.get_parameter("stagnation_decisions_threshold").value
        )
        self.escape_clearance_weight = float(self.get_parameter("escape_clearance_weight").value)
        self.escape_goal_weight = float(self.get_parameter("escape_goal_weight").value)
        self.escape_min_clearance_m = float(self.get_parameter("escape_min_clearance_m").value)
        self.bounds_padding_m = float(self.get_parameter("bounds_padding_m").value)
        self.use_map_metadata_bounds = bool(self.get_parameter("use_map_metadata_bounds").value)
        map_metadata_topic = str(self.get_parameter("map_metadata_topic").value)

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        nav_action_name = str(self.get_parameter("nav_action_name").value)

        self.model = DQN.load(self.model_path)
        self.get_logger().info(f"Loaded model: {self.model_path}")

        self.scan_msg: Optional[LaserScan] = None
        self.final_goal: Optional[tuple[float, float]] = None
        self.current_goal_handle = None
        self.result_future = None
        self.last_subgoal: Optional[tuple[float, float]] = None
        self.last_subgoal_sent_at = self.get_clock().now()
        self.last_subgoal_progress_at = self.get_clock().now()
        self.best_subgoal_dist: Optional[float] = None
        self.no_progress_replans = 0
        self.map_bounds: Optional[tuple[float, float, float, float]] = None
        self.final_goal_reached_logged = False
        self.last_decision_pose: Optional[tuple[float, float]] = None
        self.stagnant_decisions = 0
        self.step_count = 0

        goal_x = float(self.get_parameter("final_goal_x").value)
        goal_y = float(self.get_parameter("final_goal_y").value)
        if not math.isnan(goal_x) and not math.isnan(goal_y):
            self.final_goal = (goal_x, goal_y)
            self.get_logger().info(f"Using static final goal from params: {self.final_goal}")

        self.create_subscription(LaserScan, scan_topic, self.on_scan, 20)
        if self.use_goal_topic:
            self.create_subscription(PoseStamped, goal_topic, self.on_goal_pose, 10)
        if self.use_map_metadata_bounds:
            self.create_subscription(MapMetaData, map_metadata_topic, self.on_map_metadata, 5)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, nav_action_name)

        self.timer = self.create_timer(1.0 / control_hz, self.on_timer)

    def on_scan(self, msg: LaserScan) -> None:
        self.scan_msg = msg

    def on_goal_pose(self, msg: PoseStamped) -> None:
        self.final_goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.get_logger().info(
            "New final goal from topic: x=%.2f y=%.2f" % (self.final_goal[0], self.final_goal[1])
        )
        self.final_goal_reached_logged = False
        self.cancel_active_goal()

    def on_map_metadata(self, msg: MapMetaData) -> None:
        min_x = float(msg.origin.position.x)
        min_y = float(msg.origin.position.y)
        max_x = min_x + float(msg.width) * float(msg.resolution)
        max_y = min_y + float(msg.height) * float(msg.resolution)
        self.map_bounds = (min_x, max_x, min_y, max_y)
        if self.step_count == 0:
            self.get_logger().info(
                "Map bounds from /map_metadata: x=[%.2f, %.2f] y=[%.2f, %.2f]"
                % (min_x, max_x, min_y, max_y)
            )

    def get_world_bounds(self) -> tuple[float, float, float, float]:
        if self.use_map_metadata_bounds and self.map_bounds is not None:
            return self.map_bounds
        return self.world_min_x, self.world_max_x, self.world_min_y, self.world_max_y

    def get_robot_pose(self) -> Optional[Pose2D]:
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except TransformException:
            return None

        q = tf_msg.transform.rotation
        return Pose2D(
            x=float(tf_msg.transform.translation.x),
            y=float(tf_msg.transform.translation.y),
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )

    def world_to_grid(self, x_world: float, y_world: float) -> tuple[int, int]:
        min_x, max_x, min_y, max_y = self.get_world_bounds()
        x_norm = (x_world - min_x) / max(1e-6, max_x - min_x)
        y_norm = (y_world - min_y) / max(1e-6, max_y - min_y)

        gx = int(round(clamp(x_norm, 0.0, 1.0) * (self.grid_size - 1.0)))
        gy = int(round(clamp(y_norm, 0.0, 1.0) * (self.grid_size - 1.0)))
        return gx, gy

    def make_local_map_from_scan(self, scan: LaserScan, robot_yaw: float) -> np.ndarray:
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
            angle_world = angle + robot_yaw
            x_local = r * math.cos(angle_world)
            y_local = r * math.sin(angle_world)

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

    def build_observation(self, pose: Pose2D, scan: LaserScan, final_goal: tuple[float, float]) -> np.ndarray:
        local_map = self.make_local_map_from_scan(scan, pose.yaw)
        local_obs = get_local_obs(local_map, radius=self.local_obs_radius)

        robot_grid = self.world_to_grid(pose.x, pose.y)
        goal_grid = self.world_to_grid(final_goal[0], final_goal[1])

        half_window = self.obs_projection_half_window
        obs_global = []
        for ox_local, oy_local in local_obs:
            gx = robot_grid[0] - half_window + ox_local
            gy = robot_grid[1] - half_window + oy_local
            obs_global.append((gx, gy))

        force = total_force(robot_grid, goal_grid, obs_global)
        direction = normalize_force(force)

        dx = (goal_grid[0] - robot_grid[0]) / self.grid_size
        dy = (goal_grid[1] - robot_grid[1]) / self.grid_size
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

    def action_to_subgoal(self, action: int, pose: Pose2D) -> tuple[float, float, float]:
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
        if self.action_swap_xy:
            # Training grid uses (row, col), while map world is (x, y).
            # Convert model action delta from grid-space to world-space.
            mx, my = my, mx
        norm = math.hypot(float(mx), float(my))
        ux, uy = float(mx) / norm, float(my) / norm

        target_x = pose.x + self.subgoal_step_m * ux
        target_y = pose.y + self.subgoal_step_m * uy
        target_yaw = math.atan2(uy, ux)
        return target_x, target_y, target_yaw

    def step_towards_goal(self, pose: Pose2D, final_goal: tuple[float, float]) -> tuple[float, float, float]:
        return self.step_towards_goal_scaled(pose, final_goal, step_scale=1.0)

    def step_towards_goal_scaled(
        self, pose: Pose2D, final_goal: tuple[float, float], step_scale: float
    ) -> tuple[float, float, float]:
        dx = final_goal[0] - pose.x
        dy = final_goal[1] - pose.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return pose.x, pose.y, pose.yaw
        safe_scale = max(0.5, float(step_scale))
        step = min(self.subgoal_step_m * safe_scale, dist)
        ux = dx / dist
        uy = dy / dist
        target_x = pose.x + step * ux
        target_y = pose.y + step * uy
        target_x, target_y = self.clamp_subgoal_to_bounds(target_x, target_y)
        target_yaw = math.atan2(target_y - pose.y, target_x - pose.x)
        return target_x, target_y, target_yaw

    def _scan_clearance_for_heading(self, scan: LaserScan, pose_yaw: float, heading_world: float) -> float:
        if scan.angle_increment == 0.0 or len(scan.ranges) == 0:
            return 0.0
        rel = (heading_world - pose_yaw) % (2.0 * math.pi)
        idx = int(round((rel - float(scan.angle_min)) / float(scan.angle_increment)))
        idx = max(0, min(len(scan.ranges) - 1, idx))
        win = 3
        vals: list[float] = []
        for j in range(max(0, idx - win), min(len(scan.ranges), idx + win + 1)):
            r = float(scan.ranges[j])
            if math.isfinite(r):
                vals.append(min(r, self.scan_max_range))
        if not vals:
            return 0.0
        vals.sort()
        return vals[len(vals) // 2]

    def choose_escape_subgoal(self, pose: Pose2D, final_goal: tuple[float, float]) -> tuple[float, float, float]:
        if self.scan_msg is None:
            return self.step_towards_goal_scaled(pose, final_goal, self.recovery_step_scale)

        goal_heading = math.atan2(final_goal[1] - pose.y, final_goal[0] - pose.x)
        offsets = [0.0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2, 3 * math.pi / 4, -3 * math.pi / 4, math.pi]
        best = None
        for off in offsets:
            heading = goal_heading + off
            clearance = self._scan_clearance_for_heading(self.scan_msg, pose.yaw, heading)
            if clearance < self.escape_min_clearance_m:
                continue
            step = min(self.subgoal_step_m * self.recovery_step_scale, max(0.2, clearance * 0.8))
            tx = pose.x + step * math.cos(heading)
            ty = pose.y + step * math.sin(heading)
            tx, ty = self.clamp_subgoal_to_bounds(tx, ty)
            next_dist = math.hypot(final_goal[0] - tx, final_goal[1] - ty)
            score = self.escape_goal_weight * (-next_dist) + self.escape_clearance_weight * clearance
            if (best is None) or (score > best[0]):
                best = (score, tx, ty)

        if best is None:
            return self.step_towards_goal_scaled(pose, final_goal, self.recovery_step_scale)

        _, tx, ty = best
        tyaw = math.atan2(ty - pose.y, tx - pose.x)
        return tx, ty, tyaw

    def clamp_subgoal_to_bounds(self, x: float, y: float) -> tuple[float, float]:
        min_x, max_x, min_y, max_y = self.get_world_bounds()
        pad = max(0.0, self.bounds_padding_m)
        low_x = min_x + pad
        high_x = max_x - pad
        low_y = min_y + pad
        high_y = max_y - pad
        if low_x > high_x:
            low_x, high_x = min_x, max_x
        if low_y > high_y:
            low_y, high_y = min_y, max_y
        return clamp(x, low_x, high_x), clamp(y, low_y, high_y)

    def send_subgoal(self, x: float, y: float, yaw: float) -> None:
        if not self.nav_client.server_is_ready():
            self.get_logger().warn("Nav2 action server is not ready yet.")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = self.map_frame
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.on_goal_response)

        self.last_subgoal = (x, y)
        self.last_subgoal_sent_at = self.get_clock().now()
        self.last_subgoal_progress_at = self.last_subgoal_sent_at
        self.best_subgoal_dist = None

    def on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Goal response failed: {exc}")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Subgoal was rejected by Nav2.")
            self.current_goal_handle = None
            self.result_future = None
            return

        self.current_goal_handle = goal_handle
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.on_goal_result)

    def on_goal_result(self, future) -> None:
        self.current_goal_handle = None
        self.result_future = None
        try:
            result_msg = future.result()
            status = int(getattr(result_msg, "status", GoalStatus.STATUS_UNKNOWN))
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.no_progress_replans = 0
            elif status == GoalStatus.STATUS_ABORTED:
                self.no_progress_replans += 1
                self.get_logger().warn(
                    "Nav2 aborted subgoal. Increasing recovery counter to %d."
                    % self.no_progress_replans
                )
        except Exception:
            pass

    def cancel_active_goal(self) -> None:
        if self.current_goal_handle is not None:
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception:
                pass
        self.current_goal_handle = None
        self.result_future = None
        self.best_subgoal_dist = None

    def goal_is_active(self) -> bool:
        return self.current_goal_handle is not None

    def on_timer(self) -> None:
        if self.scan_msg is None:
            return

        if self.final_goal is None:
            if self.step_count % 10 == 0:
                self.get_logger().info("Waiting for final goal (params or /goal_pose).")
            self.step_count += 1
            return

        pose = self.get_robot_pose()
        if pose is None:
            return

        dist_final = math.hypot(self.final_goal[0] - pose.x, self.final_goal[1] - pose.y)
        if dist_final <= self.final_goal_tolerance:
            self.cancel_active_goal()
            if not self.final_goal_reached_logged:
                self.get_logger().info(
                    f"Final goal reached. dist={dist_final:.3f} m, decisions={self.step_count}"
                )
                self.final_goal_reached_logged = True
            return
        self.final_goal_reached_logged = False

        if self.goal_is_active():
            if self.last_subgoal is not None:
                dist_subgoal = math.hypot(self.last_subgoal[0] - pose.x, self.last_subgoal[1] - pose.y)
                now = self.get_clock().now()
                if self.best_subgoal_dist is None:
                    self.best_subgoal_dist = dist_subgoal
                    self.last_subgoal_progress_at = now
                elif dist_subgoal < (self.best_subgoal_dist - 0.03):
                    # Count as progress only if movement to subgoal is meaningful.
                    self.best_subgoal_dist = dist_subgoal
                    self.last_subgoal_progress_at = now
                    self.no_progress_replans = 0

                no_progress_elapsed = (now - self.last_subgoal_progress_at).nanoseconds / 1e9
                if dist_subgoal <= self.subgoal_tolerance_m:
                    self.cancel_active_goal()
                elif no_progress_elapsed >= self.subgoal_timeout_sec:
                    self.no_progress_replans += 1
                    self.get_logger().warn(
                        "Subgoal no-progress timeout (dist=%.2f), canceling and replanning."
                        % dist_subgoal
                    )
                    self.cancel_active_goal()
            return

        if self.last_decision_pose is None:
            self.last_decision_pose = (pose.x, pose.y)
            self.stagnant_decisions = 0
        else:
            pose_delta = math.hypot(pose.x - self.last_decision_pose[0], pose.y - self.last_decision_pose[1])
            if pose_delta < self.stagnation_dist_epsilon_m:
                self.stagnant_decisions += 1
            else:
                self.stagnant_decisions = 0
            self.last_decision_pose = (pose.x, pose.y)

        if dist_final <= self.direct_goal_radius:
            target_x = self.final_goal[0]
            target_y = self.final_goal[1]
            target_yaw = math.atan2(self.final_goal[1] - pose.y, self.final_goal[0] - pose.x)
            self.send_subgoal(target_x, target_y, target_yaw)
            self.step_count += 1
            if self.step_count % 5 == 0:
                self.get_logger().info(
                    "decision=%d direct_goal pos=(%.2f,%.2f) final_dist=%.2f subgoal=(%.2f,%.2f)"
                    % (
                        self.step_count,
                        pose.x,
                        pose.y,
                        dist_final,
                        target_x,
                        target_y,
                    )
                )
            return

        decision_mode = ""
        if self.stagnant_decisions >= self.stagnation_decisions_threshold:
            target_x, target_y, target_yaw = self.choose_escape_subgoal(pose, self.final_goal)
            decision_mode = "stagnation_escape"
        elif self.no_progress_replans >= self.recovery_timeout_replans:
            target_x, target_y, target_yaw = self.choose_escape_subgoal(pose, self.final_goal)
            decision_mode = "recovery_escape"
        else:
            obs = self.build_observation(pose, self.scan_msg, self.final_goal)
            action, _ = self.model.predict(obs, deterministic=True)
            target_x, target_y, target_yaw = self.action_to_subgoal(int(action), pose)
            target_x, target_y = self.clamp_subgoal_to_bounds(target_x, target_y)
            target_yaw = math.atan2(target_y - pose.y, target_x - pose.x)
            next_dist = math.hypot(self.final_goal[0] - target_x, self.final_goal[1] - target_y)
            if next_dist > dist_final + self.away_from_goal_margin_m or math.hypot(
                target_x - pose.x, target_y - pose.y
            ) < 0.05:
                target_x, target_y, target_yaw = self.step_towards_goal(pose, self.final_goal)
                decision_mode = f"action={int(action)}->fallback_goal"
            else:
                decision_mode = f"action={int(action)}"
        self.send_subgoal(target_x, target_y, target_yaw)

        self.step_count += 1
        if self.step_count % 5 == 0:
            self.get_logger().info(
                "decision=%d mode=%s pos=(%.2f,%.2f) final_dist=%.2f subgoal=(%.2f,%.2f)"
                % (
                    self.step_count,
                    decision_mode,
                    pose.x,
                    pose.y,
                    dist_final,
                    target_x,
                    target_y,
                )
            )


def main() -> None:
    rclpy.init()
    node = HybridNav2Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.cancel_active_goal()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
