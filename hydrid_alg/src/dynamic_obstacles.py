import math
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from rclpy.node import Node


@dataclass
class PersonConfig:
    name: str
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    speed_mps: float
    waypoints: List[Tuple[float, float]]
    color_rgba: Tuple[float, float, float, float]


@dataclass
class PersonState:
    x: float
    y: float
    wp_idx: int
    wp_next: int
    dir_sign: int  # +1 forward, -1 backward (ping-pong)


class DynamicObstaclesNode(Node):
    def __init__(self) -> None:
        super().__init__("dynamic_obstacles")

        self.declare_parameter("update_hz", 60.0)
        self.declare_parameter("spawn_service_candidates", ["/spawn_entity", "/gazebo/spawn_entity"])
        self.declare_parameter("delete_service_candidates", ["/delete_entity"])
        self.declare_parameter("set_state_service_candidates", ["/gazebo/set_entity_state", "/set_entity_state"])
        self.declare_parameter("service_wait_sec", 30.0)
        self.declare_parameter("set_state_connect_timeout_sec", 10.0)
        self.declare_parameter("set_state_retry_sec", 5.0)
        self.declare_parameter("max_yaw_rate_rad_s", 2.6)

        self.update_hz = float(self.get_parameter("update_hz").value)
        wait_sec = float(self.get_parameter("service_wait_sec").value)
        set_state_wait_sec = float(self.get_parameter("set_state_connect_timeout_sec").value)
        self._set_state_retry_sec = float(self.get_parameter("set_state_retry_sec").value)
        self._max_yaw_rate = float(self.get_parameter("max_yaw_rate_rad_s").value)
        spawn_candidates = list(self.get_parameter("spawn_service_candidates").value)
        delete_candidates = list(self.get_parameter("delete_service_candidates").value)
        set_state_candidates = list(self.get_parameter("set_state_service_candidates").value)

        self._spawn_client, spawn_name = self._connect_service(SpawnEntity, spawn_candidates, wait_sec)
        self._delete_client, delete_name = self._connect_service(DeleteEntity, delete_candidates, wait_sec)
        self._set_state_client, set_state_name = self._try_connect_service(
            SetEntityState, set_state_candidates, timeout_sec=set_state_wait_sec
        )
        if self._set_state_client is None:
            self.get_logger().warn(
                "SetEntityState service not found. Falling back to gz pose updates (may be less smooth)."
            )
        self.get_logger().info(
            f"Connected services: spawn={spawn_name}, delete={delete_name}, set_state={set_state_name}"
        )

        # Paths are tuned to user screenshots (house world top view):
        # person_room1_fb: mostly vertical between two rooms
        # person_room2_lr: left-right with short connector bends between rooms
        self._people: List[PersonConfig] = [
            PersonConfig(
                name="person_room1_down",
                center_z=0.88,
                size_x=0.42,
                size_y=0.34,
                size_z=1.76,
                speed_mps=1.0,
                waypoints=[
                    (-3.60, 4.15),
                    (-2.85, 4.15),
                    (-2.55, 4.15),
                    (-1.95, 4.15),
                    (-1.15, 4.15),
                    (-0.55, 4.15),
                ],
                color_rgba=(0.90, 0.80, 0.62, 1.0),
            ),
            PersonConfig(
                name="person_room1_down_2",
                center_z=0.88,
                size_x=0.42,
                size_y=0.34,
                size_z=1.76,
                speed_mps=1.2,
                waypoints=[
                    (-3.60, 0.55),
                    (-2.85, 0.55),
                    (-2.55, 0.55),
                    (-1.95, 0.15),
                    (-1.15, 0.15),
                    (-0.55, 0.15),
                    (-0.01, 0.55),
                    (0.45, 0.15),
                    (1.00, 0.15),
                    (1.30, 0.15),
                ],
                color_rgba=(0.90, 0.80, 0.62, 1.0),
            ),
            PersonConfig(
                name="person_room2_up",
                center_z=0.88,
                size_x=0.42,
                size_y=0.34,
                size_z=1.76,
                speed_mps=1.0,
                waypoints=[
                    (3.0, 0.99),
                    (3.0, 1.2),
                    (3.0, 1.45),
                    (3.0, 2.55),
                    (3.0, 3.55),
                    (3.0, 4.35),
                    ],
                color_rgba=(0.88, 0.78, 0.60, 1.0),
            ),
        ]

        self._state: Dict[str, PersonState] = {}
        self._last_pose_cmd: Dict[str, Tuple[float, float, float]] = {}
        self._yaw_state: Dict[str, float] = {}
        self._set_state_pending = {}
        self._set_state_retry_at = time.monotonic() + self._set_state_retry_sec
        self._pending_gz_cmds = []
        self._stale_entities = ["dyn_chair_lr", "dyn_cart_fb", "person_room1_down", "person_room2_up", "person_room1_down_2"]
        self._last_update_t = None
        self._set_state_candidates = set_state_candidates

        for p in self._people:
            for wx, wy in p.waypoints:
                if wx < -5.0 or wx > 5.0 or wy < -5.0 or wy > 5.0:
                    self.get_logger().warn(
                        f"Waypoint out of expected house bounds for {p.name}: ({wx:.2f}, {wy:.2f})"
                    )

        self._spawn_all()
        self._init_states()
        self.create_timer(1.0 / max(self.update_hz, 1e-3), self._update_positions)
        self.get_logger().info("Dynamic people active with smooth waypoint motion.")

    def _connect_service(self, srv_type, candidates: List[str], timeout_sec: float):
        start = time.monotonic()
        clients = [(name, self.create_client(srv_type, name)) for name in candidates]
        while (time.monotonic() - start) < timeout_sec:
            for name, client in clients:
                if client.wait_for_service(timeout_sec=0.25):
                    return client, name
            self.get_logger().info(f"Waiting for service among: {candidates}")
        raise RuntimeError(f"Failed to connect service. Tried: {candidates}")

    def _try_connect_service(self, srv_type, candidates: List[str], timeout_sec: float):
        start = time.monotonic()
        clients = [(name, self.create_client(srv_type, name)) for name in candidates]
        while (time.monotonic() - start) < timeout_sec:
            for name, client in clients:
                if client.wait_for_service(timeout_sec=0.2):
                    return client, name
        return None, "not_available"

    def _build_person_sdf(self, p: PersonConfig) -> str:
        r, g, b, a = p.color_rgba
        body_h = p.size_z * 0.52
        leg_h = p.size_z * 0.38
        head_r = min(p.size_x, p.size_y) * 0.30
        body_wx = p.size_x * 0.62
        body_wy = p.size_y * 0.70
        leg_wx = p.size_x * 0.22
        leg_wy = p.size_y * 0.32
        return f"""
<sdf version='1.6'>
  <model name='{p.name}'>
    <static>false</static>
    <allow_auto_disable>false</allow_auto_disable>
    <link name='body'>
      <gravity>false</gravity>
      <pose>0 0 0 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.4</ixx><ixy>0.0</ixy><ixz>0.0</ixz>
          <iyy>0.4</iyy><iyz>0.0</iyz>
          <izz>0.2</izz>
        </inertia>
      </inertial>
      <collision name='collision'>
        <pose>0 0 {p.size_z * 0.50} 0 0 0</pose>
        <geometry>
          <box><size>{p.size_x} {p.size_y} {p.size_z}</size></box>
        </geometry>
      </collision>
      <visual name='torso'>
        <pose>0 0 {leg_h + body_h * 0.5} 0 0 0</pose>
        <geometry><box><size>{body_wx} {body_wy} {body_h}</size></box></geometry>
        <material><ambient>{r} {g} {b} {a}</ambient><diffuse>{r} {g} {b} {a}</diffuse></material>
      </visual>
      <visual name='head'>
        <pose>0 0 {leg_h + body_h + head_r * 1.10} 0 0 0</pose>
        <geometry><sphere><radius>{head_r}</radius></sphere></geometry>
        <material><ambient>0.96 0.83 0.69 {a}</ambient><diffuse>0.96 0.83 0.69 {a}</diffuse></material>
      </visual>
      <visual name='leg_left'>
        <pose>{p.size_x * 0.16} 0 {leg_h * 0.5} 0 0 0</pose>
        <geometry><box><size>{leg_wx} {leg_wy} {leg_h}</size></box></geometry>
        <material><ambient>0.16 0.16 0.20 {a}</ambient><diffuse>0.16 0.16 0.20 {a}</diffuse></material>
      </visual>
      <visual name='leg_right'>
        <pose>{-p.size_x * 0.16} 0 {leg_h * 0.5} 0 0 0</pose>
        <geometry><box><size>{leg_wx} {leg_wy} {leg_h}</size></box></geometry>
        <material><ambient>0.16 0.16 0.20 {a}</ambient><diffuse>0.16 0.16 0.20 {a}</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
""".strip()

    def _delete_if_exists(self, entity_name: str) -> None:
        req = DeleteEntity.Request()
        req.name = entity_name
        fut = self._delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

    def _spawn_all(self) -> None:
        for entity_name in self._stale_entities:
            self._delete_if_exists(entity_name)
        for p in self._people:
            start_x, start_y = p.waypoints[0]
            start_x = float(start_x)
            start_y = float(start_y)
            req = SpawnEntity.Request()
            req.name = p.name
            req.xml = self._build_person_sdf(p)
            req.robot_namespace = ""
            req.reference_frame = "world"
            req.initial_pose.position.x = float(start_x)
            req.initial_pose.position.y = float(start_y)
            req.initial_pose.position.z = 0.0
            req.initial_pose.orientation.w = 1.0

            fut = self._spawn_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            resp = fut.result()
            if resp is not None and resp.success:
                self.get_logger().info(f"Spawned person: {p.name}")
            elif resp is not None and ("exist" in (resp.status_message or "").lower()):
                self.get_logger().info(f"Person already exists: {p.name}")
            else:
                self.get_logger().warn(f"Spawn issue for {p.name}")

    def _init_states(self) -> None:
        self._state.clear()
        for p in self._people:
            x0, y0 = p.waypoints[0]
            self._state[p.name] = PersonState(x=x0, y=y0, wp_idx=0, wp_next=1, dir_sign=1)

    def _advance_waypoint_indices(self, p: PersonConfig, st: PersonState) -> None:
        n = len(p.waypoints)
        if st.dir_sign > 0:
            if st.wp_next >= n - 1:
                st.dir_sign = -1
                st.wp_idx = st.wp_next
                st.wp_next = st.wp_idx - 1
            else:
                st.wp_idx = st.wp_next
                st.wp_next = st.wp_idx + 1
        else:
            if st.wp_next <= 0:
                st.dir_sign = 1
                st.wp_idx = st.wp_next
                st.wp_next = st.wp_idx + 1
            else:
                st.wp_idx = st.wp_next
                st.wp_next = st.wp_idx - 1

    def _update_positions(self) -> None:
        now = time.monotonic()
        if self._last_update_t is None:
            self._last_update_t = now
            return
        dt = max(0.005, min(0.2, now - self._last_update_t))
        self._last_update_t = now
        self._maybe_reconnect_set_state(now)
        self._pending_gz_cmds = []

        for p in self._people:
            st = self._state[p.name]
            dist_left = p.speed_mps * dt
            while dist_left > 1e-6:
                tx, ty = p.waypoints[st.wp_next]
                dx = tx - st.x
                dy = ty - st.y
                seg = math.hypot(dx, dy)
                if seg < 1e-6:
                    self._advance_waypoint_indices(p, st)
                    continue
                step = min(dist_left, seg)
                st.x += dx / seg * step
                st.y += dy / seg * step
                dist_left -= step
                if step >= seg - 1e-6:
                    self._advance_waypoint_indices(p, st)

            tx, ty = p.waypoints[st.wp_next]
            yaw_target = math.atan2(ty - st.y, tx - st.x)
            yaw_prev = self._yaw_state.get(p.name, yaw_target)
            yaw = self._smooth_yaw(yaw_prev, yaw_target, dt)
            self._yaw_state[p.name] = yaw
            self._set_pose(p.name, st.x, st.y, 0.0, yaw)
        self._flush_gz_pose_batch()

    def _maybe_reconnect_set_state(self, now: float) -> None:
        if self._set_state_client is not None:
            return
        if now < self._set_state_retry_at:
            return
        client, name = self._try_connect_service(SetEntityState, self._set_state_candidates, timeout_sec=0.5)
        if client is not None:
            self._set_state_client = client
            self.get_logger().info(f"Connected set_state service late: {name}")
        self._set_state_retry_at = now + self._set_state_retry_sec

    def _smooth_yaw(self, current: float, target: float, dt: float) -> float:
        delta = math.atan2(math.sin(target - current), math.cos(target - current))
        max_step = self._max_yaw_rate * max(dt, 1e-3)
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step
        return current + delta

    def _set_pose(self, model_name: str, x: float, y: float, z: float, yaw: float) -> None:
        prev: Optional[Tuple[float, float, float]] = self._last_pose_cmd.get(model_name)
        if prev is not None:
            if abs(prev[0] - x) < 0.001 and abs(prev[1] - y) < 0.001 and abs(prev[2] - yaw) < 0.01:
                return
        self._last_pose_cmd[model_name] = (x, y, yaw)

        if self._set_state_client is not None:
            prev_fut = self._set_state_pending.get(model_name)
            if prev_fut is not None and not prev_fut.done():
                return
            req = SetEntityState.Request()
            state = EntityState()
            state.name = model_name
            state.reference_frame = "world"
            state.pose.position.x = float(x)
            state.pose.position.y = float(y)
            state.pose.position.z = float(z)
            state.pose.orientation.z = math.sin(yaw / 2.0)
            state.pose.orientation.w = math.cos(yaw / 2.0)
            state.twist.linear.x = 0.0
            state.twist.linear.y = 0.0
            state.twist.linear.z = 0.0
            state.twist.angular.z = 0.0
            req.state = state
            self._set_state_pending[model_name] = self._set_state_client.call_async(req)
            return

        self._pending_gz_cmds.append(
            f"gz model -m {model_name} -x {x:.4f} -y {y:.4f} -z {z:.4f} -R 0.0 -P 0.0 -Y {yaw:.4f}"
        )

    def _flush_gz_pose_batch(self) -> None:
        if not self._pending_gz_cmds:
            return
        batch = " ; ".join(self._pending_gz_cmds)
        subprocess.run(["bash", "-lc", batch], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynamicObstaclesNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
