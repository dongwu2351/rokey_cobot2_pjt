import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import tf2_ros
from realsense import ImgNode
from scipy.spatial.transform import Rotation
from onrobot import RG

import time
import threading
import numpy as np


import DR_init

# for single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# --- 깊이 샘플링 ---
DEPTH_PATCH = 5        # 클릭 지점 깊이값을 중앙값으로 뽑을 창 크기 (px, 홀수)
RING_INNER_MM = 40     # 바닥면 추정용 링 안쪽 반지름 (mm, 실제 크기 기준)
RING_OUTER_MM = 70     # 바닥면 추정용 링 바깥 반지름 (mm)
RING_MIN_SAMPLES = 8   # 바닥면 추정에 필요한 최소 유효 샘플 수

# --- 파지 깊이 (mm) ---
# 링으로 추정한 물체 높이는 링이 물체 위에 걸치면 과소평가된다 (실측 19mm ->
# plunge 9.7mm). 그 깊이로는 손가락이 물체를 물지 못해서 고정값으로 되돌렸다.
# 높이 추정 자체는 놓을 높이 계산에만 계속 쓴다. 거기서 몇 mm 틀려봐야
# 살짝 떨어뜨리는 정도지만, 파지 깊이가 얕으면 아예 못 잡기 때문이다.
PLUNGE_MM = 25         # 물체 윗면에서 파고들 깊이 (고정)
ADAPTIVE_PLUNGE = False  # True 로 바꾸면 아래 비율 기반 적응형으로 동작
PLUNGE_RATIO = 0.5     # 물체 높이의 몇 %까지 파고들 것인가
PLUNGE_MIN = 5         # 최소 파지 깊이
PLUNGE_MAX = 40        # 최대 파지 깊이 (그리퍼 손가락 길이 한계)

# --- 경로 (mm) ---
APPROACH_HEIGHT = 100  # 목표 바로 위 접근/이탈 높이
PLACE_CLEARANCE = 3    # 놓을 때 바닥에서 띄우는 여유
PLACE_BLIND_DROP = 50  # 물체 높이를 모를 때 안전하게 떨어뜨릴 높이
JOG_STEP = 20          # w / x 키로 한 번에 움직일 z 거리

# --- TF 프레임 (마커를 실제 위치에 고정하는 데 사용) ---
BASE_FRAME = "base_link"
FLANGE_FRAME = "link_6"

# --- 화면 표시 색상 (BGR) ---
COLOR_PICK = (80, 220, 80)
COLOR_PLACE = (80, 160, 255)
COLOR_TEXT = (255, 255, 255)


class TestNode(Node):
    def __init__(self):
        super().__init__("test_node")

        self.img_node = ImgNode()

        # TF 로 그리퍼 자세를 받는다. 마커를 실제 물체 위에 고정(anchor)하려면
        # 매 프레임 로봇 자세가 필요한데, get_current_posx() 는 서비스라 동작
        # 중에는 워커가 로봇 노드를 독점해서 쓸 수 없다. TF 는 순수 토픽이라
        # 팔이 움직이는 동안에도 계속 들어온다.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=False
        )

        # 카메라/TF 수신 전용 스레드 + 전용 executor.
        #
        # rclpy.spin(node) 과 rclpy.spin_until_future_complete(node, fut) 은
        # executor 를 안 넘기면 둘 다 get_global_executor() 를 쓴다. 즉 이 스레드가
        # rclpy.spin 으로 전역 executor 를 붙잡고 있으면, DSR_ROBOT2 의 서비스
        # 호출(get_current_posx, movel ...)이 같은 executor 에 들어와서
        #     ValueError: generator already executing
        # 으로 터진다. 그래서 여기는 자기 executor 를 따로 갖는다.
        # (DSR 쪽은 라이브러리 코드라 전역 executor 를 계속 쓴다.)
        self._cam_exec = SingleThreadedExecutor()
        self._cam_exec.add_node(self.img_node)
        self._cam_exec.add_node(self)          # TF 구독도 이 executor 에서 돈다
        self._cam_thread = threading.Thread(target=self._cam_exec.spin, daemon=True)
        self._cam_thread.start()

        self.intrinsics = None
        while self.intrinsics is None:
            self.get_logger().info("waiting camera_info ...")
            time.sleep(0.2)
            self.intrinsics = self.img_node.get_camera_intrinsic()
        while self.img_node.get_color_frame() is None:
            self.get_logger().info("waiting color frame ...")
            time.sleep(0.2)
        while self.img_node.get_depth_frame() is None:
            self.get_logger().info("waiting depth frame ...")
            time.sleep(0.2)
        self.get_logger().info("camera ready")

        self.gripper2cam = np.load("T_gripper2camera.npy")
        self.JReady = posj([0, 0, 90, 0, 90, 0])

        # TF 의 link_6(플랜지)와 get_current_posx 가 주는 TCP 사이의 고정 오프셋을
        # 지금(정지 상태에서) 한 번만 재둔다. 이후로는 TF 만으로 TCP 자세를
        # 복원할 수 있다.
        self.flange2tcp = self._calibrate_flange_offset()

        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            self.get_logger().info(f"gripper connected: {GRIPPER_NAME}")
        except Exception as e:
            self.gripper = None
            self.get_logger().error(f"그리퍼 연결 실패 ({e}). 파지 없이 이동만 합니다.")

        # 클릭 상태: 1번째 = PICK, 2번째 = PLACE
        self.pick = None
        self.place = None
        self.status = "click PICK point"
        self.dry_run = False
        self._last_render_state = None

        # 로봇 동작은 별도 스레드에서 돌린다.
        # movel 은 내부에서 rclpy.spin_until_future_complete(g_node, ...) 로
        # 동작이 끝날 때까지 블로킹한다. 이걸 메인 스레드에서 부르면 그동안
        # imshow 가 멈춰서 화면이 얼어붙는다.
        #
        # 스레드 배치:
        #   메인   - GUI (imshow / waitKey / 마우스 콜백)
        #   카메라 - img_node spin
        #   워커   - 로봇 노드(g_node) 서비스 호출. 한 번에 하나만.
        self.lock = threading.Lock()
        self.busy = False
        self._worker = None

        # 로봇 노드(g_node)는 전역 executor 를 쓰므로 절대 두 스레드에서 동시에
        # 건드리면 안 된다. busy 플래그와 별개로 구조적으로도 막아둔다.
        # GUI 스레드는 non-blocking 으로만 잡아서 절대 대기하지 않는다.
        self.robot_lock = threading.RLock()

    # ------------------------------------------------------------------
    # 좌표 변환
    # ------------------------------------------------------------------
    def get_camera_pos(self, center_x, center_y, center_z, intrinsics):
        camera_x = (center_x - intrinsics["ppx"]) * center_z / intrinsics["fx"]
        camera_y = (center_y - intrinsics["ppy"]) * center_z / intrinsics["fy"]
        camera_z = center_z

        return (camera_x, camera_y, camera_z)

    def get_robot_pose_matrix(self, x, y, z, rx, ry, rz):
        R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def current_base2cam(self):
        """지금 이 순간의 베이스<-카메라 변환.

        eye-in-hand 라 팔이 움직이면 바뀌므로, 한 번의 클릭을 처리하는 동안에는
        이 값을 한 번만 떠서 계속 재사용해야 좌표가 서로 어긋나지 않는다.
        """
        if not self.robot_lock.acquire(blocking=False):
            return None          # 워커가 로봇을 쓰는 중. GUI 는 기다리지 않는다.
        try:
            base2gripper = self.get_robot_pose_matrix(*get_current_posx()[0])
        finally:
            self.robot_lock.release()
        return base2gripper @ self.gripper2cam

    # ------------------------------------------------------------------
    # TF 기반 자세 (동작 중에도 사용 가능)
    # ------------------------------------------------------------------
    def tf_base2flange(self):
        """TF 로 베이스<-플랜지 변환 (mm). 못 받으면 None."""
        try:
            t = self.tf_buffer.lookup_transform(
                BASE_FRAME, FLANGE_FRAME, rclpy.time.Time()
            ).transform
        except Exception:
            return None
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(
            [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        ).as_matrix()
        T[:3, 3] = [t.translation.x * 1000.0,
                    t.translation.y * 1000.0,
                    t.translation.z * 1000.0]
        return T

    def _calibrate_flange_offset(self):
        """플랜지->TCP 고정 오프셋을 정지 상태에서 한 번 측정."""
        for _ in range(25):
            T_bf = self.tf_base2flange()
            if T_bf is not None:
                break
            time.sleep(0.2)
        else:
            self.get_logger().warn(
                f"TF({BASE_FRAME}->{FLANGE_FRAME}) 를 못 받았습니다. "
                "마커 고정 기능 없이 동작합니다."
            )
            return None

        T_bt = self.get_robot_pose_matrix(*get_current_posx()[0])
        offset = np.linalg.inv(T_bf) @ T_bt
        self.get_logger().info(
            f"flange->TCP 오프셋 {np.round(offset[:3, 3], 2)} mm"
        )
        return offset

    def tf_base2cam(self):
        """TF 로 베이스<-카메라 변환. 마커를 화면에 고정할 때 쓴다."""
        if self.flange2tcp is None:
            return None
        T_bf = self.tf_base2flange()
        if T_bf is None:
            return None
        return T_bf @ self.flange2tcp @ self.gripper2cam

    def project_to_pixel(self, p_base, base2cam):
        """베이스 좌표 -> 현재 화면 픽셀. 카메라 뒤/화면 밖이면 None."""
        P = np.linalg.inv(base2cam) @ np.append(np.asarray(p_base, float), 1.0)
        if not np.isfinite(P).all() or P[2] <= 1.0:
            return None
        u = self.intrinsics["fx"] * P[0] / P[2] + self.intrinsics["ppx"]
        v = self.intrinsics["fy"] * P[1] / P[2] + self.intrinsics["ppy"]
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return int(round(u)), int(round(v))

    def pixel_to_base(self, u, v, z, base2cam):
        """픽셀 (u,v) + 깊이 z -> 로봇 베이스 좌표 (mm)."""
        cam = self.get_camera_pos(u, v, z, self.intrinsics)
        coord = np.append(np.array(cam), 1)
        return (base2cam @ coord)[:3]

    # ------------------------------------------------------------------
    # 깊이 샘플링
    # ------------------------------------------------------------------
    def sample_depth(self, u, v, depth_frame, patch=DEPTH_PATCH):
        """(u,v) 주변 창에서 0이 아닌 깊이값의 중앙값.

        픽셀 하나만 읽으면 깊이 구멍(0)에 걸렸을 때 카메라 원점이 목표가 되어
        로봇이 엉뚱한 곳으로 간다. 중앙값을 쓰면 구멍과 튀는 값에 강해진다.
        """
        h, w = depth_frame.shape
        r = patch // 2
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        if y0 >= y1 or x0 >= x1:
            return None
        vals = depth_frame[y0:y1, x0:x1]
        vals = vals[vals > 0]
        if vals.size == 0:
            return None
        return float(np.median(vals))

    def estimate_floor_height(self, u, v, depth_frame, base2cam, center_depth):
        """클릭 지점을 둘러싼 링에서 바닥면의 베이스 z (mm) 를 추정.

        물체 바깥쪽 도넛 모양 영역을 훑어서 그 중앙값을 바닥으로 본다.
        링 반지름은 mm 로 정의하고 매번 픽셀로 환산한다. 픽셀로 고정해두면
        카메라가 멀어질수록 링이 실제로는 좁아져서 물체 위를 훑게 된다.
            r_px = r_mm * fx / z
        """
        fx = self.intrinsics["fx"]
        r_in = RING_INNER_MM * fx / center_depth
        r_out = RING_OUTER_MM * fx / center_depth

        heights = []
        for ang in range(0, 360, 15):
            c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            for r in (r_in, (r_in + r_out) / 2, r_out):
                uu, vv = int(u + r * c), int(v + r * s)
                d = self.sample_depth(uu, vv, depth_frame, patch=3)
                if d is None:
                    continue
                heights.append(self.pixel_to_base(uu, vv, d, base2cam)[2])

        if len(heights) < RING_MIN_SAMPLES:
            return None
        return float(np.median(heights))

    def compute_plunge(self, height):
        """파고들 깊이 (mm). 기본은 고정값."""
        if not ADAPTIVE_PLUNGE or height is None or height <= 0:
            return float(PLUNGE_MM)
        return float(np.clip(height * PLUNGE_RATIO, PLUNGE_MIN, PLUNGE_MAX))

    # ------------------------------------------------------------------
    # 클릭 처리
    # ------------------------------------------------------------------
    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.busy:
            return

        depth_frame = self.img_node.get_depth_frame()
        if depth_frame is None or np.all(depth_frame == 0):
            self.status = "no depth frame yet"
            return

        base2cam = self.current_base2cam()
        if base2cam is None:
            self.status = "robot busy - try again"
            return

        d = self.sample_depth(x, y, depth_frame)
        if d is None:
            self.status = "no depth at that pixel - click again"
            self.get_logger().warn(f"({x}, {y}) 깊이값 없음. 다시 클릭하세요.")
            return

        pos = self.pixel_to_base(x, y, d, base2cam)
        floor_z = self.estimate_floor_height(x, y, depth_frame, base2cam, d)

        print(f"img cordinate: ({x}, {y})  depth: {d:.1f} mm")
        print(f"robot cordinate: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        if self.pick is None:
            height = None if floor_z is None else pos[2] - floor_z
            if height is not None and height <= 0:
                height = None
            plunge = self.compute_plunge(height)

            with self.lock:
                self.pick = {"uv": (x, y), "pos": pos,
                             "height": height, "plunge": plunge}
            if height is None:
                self.get_logger().warn(
                    f"물체 높이 추정 실패 -> 고정 {plunge:.0f}mm 사용"
                )
                self.status = "PICK set (height unknown) - click PLACE point"
            else:
                print(f"object height: {height:.1f} mm -> plunge {plunge:.1f} mm")
                self.status = (
                    f"PICK set  h={height:.0f}mm plunge={plunge:.0f}mm"
                    " - click PLACE point"
                )
        else:
            with self.lock:
                self.place = {"uv": (x, y), "pos": pos, "surface_z": float(pos[2])}
            # 여기서 바로 pick_and_place() 를 부르면 화면이 멈춘다. 워커로 넘긴다.
            self.start_job(self._job_pick_and_place)

    # ------------------------------------------------------------------
    # 워커 스레드
    # ------------------------------------------------------------------
    def start_job(self, fn):
        """로봇 동작을 워커 스레드에서 실행. 이미 동작 중이면 무시."""
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        self._worker = threading.Thread(target=self._run_job, args=(fn,), daemon=True)
        self._worker.start()
        return True

    def _run_job(self, fn):
        try:
            with self.robot_lock:      # 로봇 노드는 이 스레드가 독점한다
                fn()
        except Exception as e:
            self.get_logger().error(f"동작 중단: {e}")
            self.status = "FAILED - see console"
        finally:
            with self.lock:
                self.busy = False

    def _job_pick_and_place(self):
        try:
            self.pick_and_place()
            self.status = "done - click PICK point"
        finally:
            with self.lock:
                self.pick = None
                self.place = None
            print("=" * 100)

    # ------------------------------------------------------------------
    # 동작
    # ------------------------------------------------------------------
    def go(self, label, x, y, z, rx, ry, rz):
        """직선 이동 + 결과 검증.

        movel 은 실패해도 예외를 던지지 않고 -1 을 반환만 한다. 반환값을
        확인하지 않으면 PICK 이동이 실패해도 조용히 PLACE 까지 진행해버린다.
        """
        target = [float(x), float(y), float(z), rx, ry, rz]
        print(f"  [{label:18s}] 명령 ({x:8.1f}, {y:8.1f}, {z:8.1f})")

        if self.dry_run:
            print(f"  {'':18s}   DRY RUN - 움직이지 않음")
            return

        ret = movel(posx(target), vel=VELOCITY, acc=ACC)
        if ret != 0:
            raise RuntimeError(
                f"[{label}] movel 실패 (ret={ret}). "
                f"목표 ({x:.1f}, {y:.1f}, {z:.1f}) 가 작업영역 밖이거나 "
                f"로봇이 정지 상태일 수 있습니다."
            )

        actual = get_current_posx()[0]
        err = float(np.linalg.norm(np.array(actual[:3]) - np.array(target[:3])))
        print(f"  {'':18s}   도달 ({actual[0]:8.1f}, {actual[1]:8.1f},"
              f" {actual[2]:8.1f})  오차 {err:5.1f} mm")
        if err > 5.0:
            self.get_logger().warn(f"[{label}] 목표와 {err:.1f}mm 차이")

    def grip(self, close):
        if self.gripper is None:
            print(f"  [{'gripper':18s}] 연결 안 됨 - 건너뜀")
            return
        if self.dry_run:
            print(f"  [{'gripper':18s}] DRY RUN - {'close' if close else 'open'}")
            return
        if close:
            self.gripper.close_gripper()
        else:
            self.gripper.open_gripper()
        time.sleep(1.0)

    def pick_and_place(self):
        p = self.pick["pos"]
        plunge = self.pick["plunge"]
        height = self.pick["height"]
        q = self.place["pos"]

        current_pos = get_current_posx()[0]
        rx, ry, rz = current_pos[3], current_pos[4], current_pos[5]

        # 물체를 놓을 때 TCP 높이.
        # 파지 후 TCP 는 물체 윗면보다 plunge 만큼 아래에 있고, 물체 바닥은
        # 거기서 다시 (height - plunge) 만큼 아래에 있다. 그 값을 되돌려준다.
        if height is None:
            place_z = self.place["surface_z"] + PLACE_BLIND_DROP
            self.get_logger().warn(
                f"물체 높이를 몰라 {PLACE_BLIND_DROP}mm 위에서 떨어뜨립니다."
            )
        else:
            place_z = self.place["surface_z"] + (height - plunge) + PLACE_CLEARANCE

        print("-" * 100)
        print(f"PICK  ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})  "
              f"height={'?' if height is None else f'{height:.1f}'}mm  "
              f"plunge={plunge:.1f}mm")
        print(f"PLACE ({q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f})  TCP z={place_z:.1f}mm")
        print(f"자세 유지 rx,ry,rz = ({rx:.1f}, {ry:.1f}, {rz:.1f})"
              f"{'   [DRY RUN]' if self.dry_run else ''}")
        print("-" * 100)

        self.status = "opening gripper"
        self.grip(close=False)

        # --- PICK ---
        self.status = "approaching PICK"
        self.go("pick approach", p[0], p[1], p[2] + APPROACH_HEIGHT, rx, ry, rz)
        self.status = f"descending {plunge:.0f}mm"
        self.go("pick descend", p[0], p[1], p[2] - plunge, rx, ry, rz)
        self.status = "grasping"
        self.grip(close=True)
        self.status = "lifting"
        self.go("pick lift", p[0], p[1], p[2] + APPROACH_HEIGHT, rx, ry, rz)

        # --- PLACE ---
        self.status = "approaching PLACE"
        self.go("place approach", q[0], q[1], place_z + APPROACH_HEIGHT, rx, ry, rz)
        self.status = "lowering"
        self.go("place lower", q[0], q[1], place_z, rx, ry, rz)
        self.status = "releasing"
        self.grip(close=False)
        self.status = "retreating"
        self.go("place retreat", q[0], q[1], place_z + APPROACH_HEIGHT, rx, ry, rz)

    def jog_z(self, delta):
        """현재 위치에서 베이스 z 방향으로 delta mm 이동."""
        self.status = f"jog z {delta:+.0f}mm"
        print(f"[jog] z {delta:+.0f} mm")
        if self.dry_run:
            print("      DRY RUN - 움직이지 않음")
        else:
            ret = movel(posx([0, 0, float(delta), 0, 0, 0]),
                        vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
            if ret != 0:
                raise RuntimeError(f"jog 실패 (ret={ret})")
            now = get_current_posx()[0]
            print(f"      현재 ({now[0]:.1f}, {now[1]:.1f}, {now[2]:.1f})")
        self.status = "click PICK point" if self.pick is None else self.status

    def go_home(self):
        self.status = "moving home"
        print("[home] JReady 로 이동")
        if self.dry_run:
            print("      DRY RUN - 움직이지 않음")
        else:
            ret = movej(self.JReady, vel=VELOCITY, acc=ACC)
            if ret != 0:
                raise RuntimeError(f"movej 실패 (ret={ret})")
        self.status = "click PICK point"

    # ------------------------------------------------------------------
    # 화면 표시
    # ------------------------------------------------------------------
    def marker_uv(self, entry, base2cam, shape):
        """마커를 그릴 픽셀 좌표.

        base2cam 이 있으면 저장해둔 '베이스 좌표'를 현재 카메라로 역투영해서
        실제 물체 위에 고정한다. 카메라가 그리퍼에 붙어 있으므로 팔이 움직이면
        화면 속 위치는 바뀌지만 물체는 그대로다. TF 를 못 받으면 클릭 당시의
        픽셀을 그대로 쓴다(고정 안 됨).
        """
        if base2cam is None:
            return entry["uv"], False
        uv = self.project_to_pixel(entry["pos"], base2cam)
        if uv is None:
            return None, True
        h, w = shape[:2]
        if not (0 <= uv[0] < w and 0 <= uv[1] < h):
            return None, True
        return uv, True

    def draw_marker(self, img, uv, color, label):
        u, v = uv
        cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 26, 2)
        cv2.circle(img, (u, v), 14, color, 2)
        cv2.putText(img, label, (u + 20, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    def render(self):
        # 수신은 카메라 스레드가 하고 있다. 여기선 최신 프레임만 가져다 그린다.
        img = self.img_node.get_color_frame()
        if img is None:
            return

        # 워커가 동작을 마치면서 pick/place 를 지울 수 있으므로 스냅샷을 뜬다.
        with self.lock:
            pick, place, busy = self.pick, self.place, self.busy

        # 저장해둔 베이스 좌표를 지금 카메라 위치 기준으로 다시 화면에 투영한다.
        # 팔이 움직여도 마커가 실제 물체 위에 그대로 붙어 있게 된다.
        base2cam = self.tf_base2cam()
        pick_uv = anchored = None
        place_uv = None
        if pick is not None:
            pick_uv, anchored = self.marker_uv(pick, base2cam, img.shape)
        if place is not None:
            place_uv, anchored = self.marker_uv(place, base2cam, img.shape)

        # 루프는 waitKey(1) 덕에 초당 수백~수천 번 돈다. 같은 프레임을 그때마다
        # 다시 복사/렌더하면 CPU 만 태우고 화면은 그대로다. 새 프레임이 왔을
        # 때와 상태가 바뀌었을 때만 다시 그린다.
        state = (id(img), self.status, self.dry_run, busy, pick_uv, place_uv)
        if state == self._last_render_state:
            return
        self._last_render_state = state

        vis = img.copy()

        if pick is not None and pick_uv is not None:
            p = pick["pos"]
            h = pick["height"]
            htxt = "?" if h is None else f"{h:.0f}"
            self.draw_marker(vis, pick_uv, COLOR_PICK, "PICK")
            cv2.putText(
                vis,
                f"h={htxt}mm  plunge={pick['plunge']:.0f}mm"
                f"  ({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})",
                (pick_uv[0] + 20, pick_uv[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PICK, 1, cv2.LINE_AA,
            )

        if place is not None and place_uv is not None:
            q = place["pos"]
            self.draw_marker(vis, place_uv, COLOR_PLACE, "PLACE")
            cv2.putText(
                vis,
                f"({q[0]:.0f},{q[1]:.0f},{q[2]:.0f})",
                (place_uv[0] + 20, place_uv[1] + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PLACE, 1, cv2.LINE_AA,
            )
            if pick_uv is not None:
                cv2.arrowedLine(vis, pick_uv, place_uv,
                                COLOR_PLACE, 2, tipLength=0.05)

        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(vis, self.status, (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(vis, "[w/x] z jog  [c] clear  [h] home  [d] dry  [ESC] quit",
                    (vis.shape[1] - 470, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        if self.dry_run:
            cv2.rectangle(vis, (0, vis.shape[0] - 28),
                          (vis.shape[1], vis.shape[0]), (0, 60, 60), -1)
            cv2.putText(vis, "DRY RUN - robot will NOT move",
                        (10, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (120, 255, 255), 1, cv2.LINE_AA)

        if busy:
            cv2.rectangle(vis, (0, 30), (vis.shape[1], 58), (0, 40, 90), -1)
            note = ("MOVING - clicks ignored (markers anchored to world)"
                    if anchored else
                    "MOVING - clicks ignored (no TF, markers are stale)")
            cv2.putText(vis, note, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (120, 200, 255), 1, cv2.LINE_AA)

        cv2.imshow("Webcam", vis)


if __name__ == "__main__":
    rclpy.init()
    node = rclpy.create_node("dsr_example_demo_py", namespace=ROBOT_ID)

    DR_init.__dsr__node = node

    try:
        from DSR_ROBOT2 import (
            get_current_posx,
            movej,
            movel,
            DR_MV_MOD_REL,
        )

        from DR_common2 import posx, posj

    except ImportError as e:
        print(f"Error importing DSR_ROBOT2 : {e}")
        exit(True)

    cv2.namedWindow("Webcam")

    test_node = TestNode()
    cv2.setMouseCallback("Webcam", test_node.mouse_callback)

    print()
    print("=" * 60)
    print(" 좌클릭 1회차 -> PICK,  2회차 -> PLACE 후 자동 실행")
    print(f" w / x : z 축 위 / 아래로 {JOG_STEP}mm")
    print(" c : 클릭 초기화   h : 홈 위치   d : DRY RUN 토글   ESC : 종료")
    print("=" * 60)
    print()

    while True:
        test_node.render()

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            if test_node.busy:
                print("동작 중입니다. 끝난 뒤 다시 ESC 를 누르세요.")
                continue
            break
        elif key == ord("c"):  # 클릭 초기화
            if not test_node.busy:
                with test_node.lock:
                    test_node.pick = None
                    test_node.place = None
                test_node.status = "click PICK point"
        elif key == ord("h"):  # 홈 위치로
            test_node.start_job(test_node.go_home)
        elif key == ord("w"):  # z 위로
            test_node.start_job(lambda: test_node.jog_z(+JOG_STEP))
        elif key == ord("x"):  # z 아래로
            test_node.start_job(lambda: test_node.jog_z(-JOG_STEP))
        elif key == ord("d"):  # DRY RUN 토글
            test_node.dry_run = not test_node.dry_run
            print(f"[DRY RUN] {'ON - 로봇이 움직이지 않습니다' if test_node.dry_run else 'OFF'}")

    cv2.destroyAllWindows()
    # 카메라 executor 를 먼저 멈추고 내려간다.
    # 이걸 안 하면 인터프리터가 종료될 때 abort 메시지가 뜬다.
    test_node._cam_exec.shutdown()
    rclpy.shutdown()
