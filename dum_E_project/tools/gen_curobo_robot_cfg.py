#!/usr/bin/env python3
"""
cuRobo 용 M0609 robot_cfg (m0609.yml) 생성기.

재료:
  - dsr_description2 의 xacro  -> 평면화한 URDF (메시 경로를 절대경로로 치환)
  - collision_spheres.yaml     -> gen_collision_spheres.py 가 만든 83개 구
  - SRDF 에서 뽑은 self_collision_ignore
  - URDF 관절 한계

구를 다시 생성했으면 이 스크립트도 다시 돌리면 된다.

실행:  ~/curobo_env/bin/python gen_curobo_robot_cfg.py
"""
import re
import subprocess
from pathlib import Path

import yaml

HERE = Path(__file__).parent
DESC = Path("/home/rokey/cobot_ws/src/doosan-robot2/dsr_description2")
XACRO = DESC / "xacro" / "m0609.urdf.xacro"
URDF_OUT = HERE / "m0609_flat.urdf"
CFG_OUT = HERE / "m0609.yml"
SPHERES = HERE / "collision_spheres.yaml"

LINKS = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
JOINTS = [f"joint_{i}" for i in range(1, 7)]

# ---------------------------------------------------------------------------
# 실기에서 측정한 값 (query_robot_tool.py 로 읽고 cuRobo FK 와 비교해서 산출)
#   컨트롤러 TCP 이름 : GripperDA_v1
#   측정 자세         : joint = [0, -0.02, 90.06, 0.02, 90.0, 0] deg
#   flange->TCP       : x=1.79  y=-1.07  z=238.13 mm
#   x,y 는 URDF-실기 기준오차(2.04mm) 안이라 노이즈로 보고 0 으로 둔다.
# ---------------------------------------------------------------------------
TCP_OFFSET_M = [0.0, 0.0, 0.23813]

# ---------------------------------------------------------------------------
# 그리퍼 충돌 구 (link_6 좌표계, m). link_6 로컬 +z 가 플랜지 바깥 방향.
#
# OnRobot RG2 공식 데이터시트:
#   전체 길이(마운트면->핑거팁) 213,  본체 높이 132,  마운트 폭 75,
#   본체 깊이 36,  최대 개구 110(안쪽)/124(바깥),  손가락 두께 24   [mm]
#
# 실측 flange->TCP 238.13 - RG2 213 = 25.1mm  -> 툴 체인저 두께로 설명된다.
#   z=0~25    툴 체인저
#   z=25~157  RG2 본체 (단면 75 x 36 -> 감싸는 반지름 sqrt(37.5^2+18^2)=41.6mm)
#   z=157~238 손가락
# ---------------------------------------------------------------------------
TOOLCHANGER_LEN = 0.025
RG2_BODY_LEN = 0.132
RG2_BODY_R = 0.042      # 75x36 단면의 반대각
TOOLCHANGER_R = 0.038
FINGER_R = 0.035        # 아래 주석 참고


def gripper_spheres():
    """툴체인저 + RG2 본체 + 손가락을 z 축을 따라 감싼다."""
    out = []
    z_body0 = TOOLCHANGER_LEN
    z_body1 = z_body0 + RG2_BODY_LEN
    z_tcp = TCP_OFFSET_M[2]

    # 툴 체인저
    out.append({"center": [0.0, 0.0, round(z_body0 / 2, 4)], "radius": TOOLCHANGER_R})

    # RG2 본체 — 반지름 간격으로 4개
    for i in range(4):
        z = z_body0 + (i + 0.5) * (z_body1 - z_body0) / 4
        out.append({"center": [0.0, 0.0, round(z, 4)], "radius": RG2_BODY_R})

    # 손가락 구간
    #
    # 주의: 손가락은 움직이지만 cuRobo 의 링크 구는 고정이다.
    # 완전히 벌리면 바깥 폭이 124mm(반지름 62mm)까지 가는데, 그 값으로 감싸면
    # 그리퍼가 지나치게 뚱뚱해져서 좁은 곳에 아예 접근을 못 한다.
    # 여기서는 닫힘~중간 개구 기준(FINGER_R)으로 잡는다. 완전 개방 상태는
    # 파지 직전 저속 접근 구간에서만 발생하므로 그때는 별도로 다룬다.
    for i in range(3):
        z = z_body1 + (i + 0.5) * (z_tcp - z_body1) / 3
        out.append({"center": [0.0, 0.0, round(z, 4)], "radius": FINGER_R})

    return out

# SRDF(dsr.srdf) 의 disable_collisions 를 그대로 옮긴 것.
# link_4~link_6 는 일부러 빼 둔다 — J5 극단 각도에서 실제로 부딪힐 수 있다.
SELF_COLLISION_IGNORE = {
    "base_link": ["link_1", "link_2"],
    "link_1": ["link_2", "link_3"],
    "link_2": ["link_3"],
    "link_3": ["link_4", "link_5", "link_6"],
    "link_4": ["link_5"],
    "link_5": ["link_6", "attached_object"],
    "link_6": ["attached_object"],
}


# 관절 한계를 현실적인 값으로 조인다.
#
# URDF 원본은 J1/J2/J4/J5/J6 가 +-2pi(+-360도) 다. 그대로 두면 IK 가
# "한 바퀴 돌아간" 해를 내놓을 수 있고, 그러면 TCP 가 로봇 주위를 크게
# 원을 그리며 도는 경로가 나온다. 실기에서도 케이블 때문에 그렇게 못 돈다.
JOINT_LIMITS = {                    # (lower, upper) [rad]
    # +-120. 큰 원을 그리던 원인은 null_space_weight=0 이었고 그건 0.3 으로
    # 되돌렸다. J1 한계는 계획시간/경로품질에 영향이 없었다(측정: 90/120/180
    # 모두 ~370ms, 배율 1.2x). 케이블 여유를 고려해 적당히 조여만 둔다.
    "joint_1": (-2.0944, 2.0944),
    "joint_2": (-3.14159, 3.14159),
    "joint_3": (-2.61799, 2.61799),   # 원본이 이미 이 값
    "joint_4": (-3.14159, 3.14159),
    "joint_5": (-3.14159, 3.14159),
    "joint_6": (-3.14159, 3.14159),
}


def clamp_limits(urdf: str) -> str:
    """<joint name="joint_N"> 블록의 <limit lower/upper> 를 교체한다."""
    for jn, (lo, hi) in JOINT_LIMITS.items():
        pat = re.compile(
            r'(<joint\s+name="' + jn + r'".*?<limit[^>]*?)lower="[^"]*"\s+upper="[^"]*"',
            re.S)
        urdf, n = pat.subn(rf'\g<1>lower="{lo}" upper="{hi}"', urdf, count=1)
        if n == 0:
            print(f"  ! {jn} 한계를 못 찾음 — URDF 형식 확인 필요")
    return urdf


def build_urdf() -> None:
    """xacro -> URDF. package:// 를 절대경로로 바꿔서 메시 로딩이 되게 한다."""
    out = subprocess.run(
        ["xacro", str(XACRO), "color:=white", "model:=m0609"],
        capture_output=True, text=True, check=True,
    ).stdout
    out = out.replace("package://dsr_description2/", f"file://{DESC}/")
    out = clamp_limits(out)
    # cuRobo 는 충돌 판정에 구만 쓴다. 메시는 시각화용이라 없어도 되지만
    # 파서가 열어보려 하므로 경로는 유효해야 한다.
    URDF_OUT.write_text(out)
    n = len(re.findall(r"<link\b", out))
    print(f"URDF 생성: {URDF_OUT}  (link {n}개)")


def main():
    build_urdf()
    spheres = yaml.safe_load(SPHERES.read_text())["collision_spheres"]
    # 그리퍼는 별도 링크가 아니라 link_6 에 붙여 둔다 (플랜지와 같이 움직이므로).
    g = gripper_spheres()
    spheres["link_6"] = spheres["link_6"] + g
    n_sph = sum(len(v) for v in spheres.values())

    cfg = {
        "robot_cfg": {
            "kinematics": {
                "format_version": "2.0",
                "urdf_path": str(URDF_OUT),
                "asset_root_path": str(DESC),
                "base_link": "base_link",
                # 플랜지(link_6)가 아니라 실제 TCP 를 목표 프레임으로 쓴다.
                # link_6 로 두면 "여기 놓아라"가 그리퍼 238mm 만큼 어긋난다.
                "tool_frames": ["tcp"],

                # 충돌 검사 대상. attached_object 가 파지한 물체 자리다.
                "collision_link_names": LINKS + ["attached_object"],
                "mesh_link_names": LINKS,

                # 환경 장애물에 대한 여유. 자기충돌 여유와 별개로 잡는다.
                "collision_sphere_buffer": 0.005,

                # 자기충돌 여유는 작게 유지해야 한다.
                # link_2~link_4 최소여유가 11mm 라, 이보다 크게 잡으면
                # 팔을 접는 자세가 통째로 막힌다. (check_self_collision.py 참고)
                "self_collision_buffer": {lk: 0.0 for lk in LINKS}
                | {"attached_object": 0.0},
                "self_collision_ignore": SELF_COLLISION_IGNORE,

                # 파지한 물체용으로 구 8개를 미리 잡아둔다.
                # 길쭉한 물체는 구 여러 개로 쪼개야 하므로 넉넉히.
                "extra_collision_spheres": {"attached_object": 8},
                "extra_links": {
                    # 실제 TCP (그리퍼 핑거팁). 계획 목표는 이 프레임 기준.
                    "tcp": {
                        "parent_link_name": "link_6",
                        "link_name": "tcp",
                        "joint_name": "tcp_joint",
                        "joint_type": "FIXED",
                        "fixed_transform": TCP_OFFSET_M + [1.0, 0.0, 0.0, 0.0],
                    },
                    "attached_object": {
                        "parent_link_name": "link_6",
                        "link_name": "attached_object",
                        "joint_name": "attach_joint",
                        "joint_type": "FIXED",
                        # 파지한 물체는 TCP 위치에 붙는다 (실측 238.13mm)
                        "fixed_transform": TCP_OFFSET_M + [1.0, 0.0, 0.0, 0.0],
                    }
                },

                "cspace": {
                    "joint_names": JOINTS,
                    "cspace_distance_weight": [1.0] * 6,
                    # null_space_weight 는 IK 해를 default_joint_position 쪽으로
                    # 끌어당긴다. 0 으로 두면 선호도가 사라져서 아무 해나 고르고,
                    # J1 이 크게 감긴 해가 뽑혀 TCP 가 큰 원을 그린다.
                    # 약하게 남겨두는 게 맞다.
                    "null_space_weight": [0.3] * 6,
                    # TODO 실기에서 튜닝. 지금은 보수적으로.
                    "max_acceleration": 8.0,
                    "max_jerk": 300.0,
                    # 특이점(팔 완전히 펴짐)을 피한 기본 자세
                    "default_joint_position": [0.0, 0.0, 1.57, 0.0, 1.57, 0.0],
                },
                "collision_spheres": spheres,
            }
        }
    }

    CFG_OUT.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None))
    print(f"robot_cfg 생성: {CFG_OUT}")
    print(f"  링크 {len(LINKS)}개 / 구 {n_sph}개 / attached_object 슬롯 8개")


if __name__ == "__main__":
    main()
