#!/usr/bin/env bash
# =============================================================================
#  cuRobo 설치 — 처음부터 끝까지 한 번에.
#
#  전제: sudo apt install -y python3.10-venv   (먼저 실행해 두세요)
#  실행: bash install_curobo.sh
#
#  설계 원칙
#   - venv 를 source 하지 않고 $VENV/bin/python 을 직접 호출한다.
#     (스크립트 안에서 activate 는 서브셸을 벗어나면 사라져서 신뢰할 수 없음)
#   - numpy 는 constraints 로 1.x 에 못박는다. ROS 2 Humble 의 cv_bridge 가
#     numpy 1.x 로 컴파일돼 있어서 2.x 가 들어오면 전부 import 실패한다.
# =============================================================================
set -euo pipefail

CUROBO_SRC="$HOME/curobo"
VENV="$HOME/curobo_env"
CONSTRAINTS="$VENV/constraints.txt"

ok()   { echo -e "  \033[32m✓\033[0m $*"; }
info() { echo -e "\n\033[1m$*\033[0m"; }
die()  { echo -e "  \033[31m✗ $*\033[0m"; exit 1; }

# ROS 의 setup.bash 는 미정의 변수를 참조한다. set -u 와 충돌하므로 잠시 끈다.
source_ros() { set +u; source /opt/ros/humble/setup.bash; set -u; }

# -----------------------------------------------------------------------------
info "[0/7] 사전 점검"
dpkg -s python3.10-venv &>/dev/null \
  || die "python3.10-venv 미설치.  sudo apt install -y python3.10-venv  먼저 실행하세요"
ok "python3.10-venv"

command -v nvcc &>/dev/null || die "nvcc 없음. CUDA_HOME 확인 필요"
NVCC_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
[ "$NVCC_VER" = "12.4" ] || echo "  ! nvcc $NVCC_VER (12.4 기대). torch 와 맞는지 확인하세요"
ok "nvcc $NVCC_VER"

[ -d /opt/ros/humble ] || die "ROS 2 Humble 없음"
ok "ROS 2 Humble"

# -----------------------------------------------------------------------------
info "[1/7] 이전 설치가 시스템 환경을 오염시켰다면 되돌리기"
# cuRobo 가 딸려 끌고 온 패키지들. 시스템 python 에 있으면 안 된다.
# (pygments/psutil/attrs/rich 등 범용 패키지는 다른 것이 쓸 수 있으므로 건드리지 않는다)
python3 -m pip uninstall -y -q \
    nvidia-curobo warp-lang viser tyro yourdfpy \
    numpy-quaternion plyfile vcs-versioning vhacdx \
    manifold3d mapbox_earcut embreex pycollada svg.path \
    rtree shapely scikit-image tifffile imageio lazy-loader \
    colorlog msgspec typeguard docstring-parser setuptools_scm &>/dev/null || true

python3 -m pip install -q --force-reinstall \
    "numpy==1.24.4" "setuptools==75.6.0" "packaging==25.0" 2>&1 \
    | grep -v "dependency resolver" || true
ok "시스템 numpy/setuptools/packaging 복원"

python3 -c "from cv_bridge import CvBridge" 2>/dev/null \
  && ok "cv_bridge 정상" \
  || echo "  ! cv_bridge import 실패 — sudo apt install --reinstall ros-humble-cv-bridge 필요할 수 있음"
python3 -c "import colcon_core" 2>/dev/null \
  && ok "colcon 정상" \
  || echo "  ! colcon import 실패"

# -----------------------------------------------------------------------------
info "[2/7] venv 새로 생성"
rm -rf "$VENV"
source_ros
python3 -m venv --system-site-packages "$VENV"
[ -x "$VENV/bin/python" ] || die "venv 생성 실패"
ok "$VENV"

# -----------------------------------------------------------------------------
info "[3/7] venv 안에서 rclpy / torch 보이는지 확인"
"$VENV/bin/python" - <<'PY' || die "게이트 실패 — rclpy 또는 torch 를 못 봅니다"
import torch, rclpy
assert torch.cuda.is_available(), "torch 가 CUDA 를 못 봅니다"
print(f"  torch {torch.__version__} (cuda {torch.version.cuda}) | rclpy OK")
PY
ok "게이트 통과"

# -----------------------------------------------------------------------------
info "[4/7] numpy 1.x 를 venv 안에 못박기"
cat > "$CONSTRAINTS" <<'EOF'
numpy<2
setuptools<80
packaging<26
EOF
# setuptools>=64 가 필수. venv 는 ensurepip 로 59.6.0 을 심는데, 그 버전은
# PEP 660(build_editable) 을 지원하지 않아 -e 설치가 실패한다.
# setuptools_scm 은 cuRobo 의 빌드 의존성 (--no-build-isolation 이라 미리 있어야 함).
"$VENV/bin/pip" install -q --upgrade \
    pip wheel ninja "setuptools>=64,<80" setuptools_scm -c "$CONSTRAINTS"
ok "setuptools $("$VENV/bin/python" -c 'import setuptools;print(setuptools.__version__)') (venv 내부)"
# --ignore-installed: 시스템에 numpy 가 있어도 venv 안에 자기 사본을 갖게 한다
"$VENV/bin/pip" install -q --ignore-installed "numpy==1.24.4"
ok "numpy $("$VENV/bin/python" -c 'import numpy;print(numpy.__version__)') (venv 내부)"

# -----------------------------------------------------------------------------
info "[5/7] cuRobo 소스 준비"
if [ ! -d "$CUROBO_SRC/.git" ]; then
    git clone https://github.com/NVlabs/curobo.git "$CUROBO_SRC"
fi
ok "$CUROBO_SRC"

# -----------------------------------------------------------------------------
info "[6/7] cuRobo 빌드 (10~20분 소요)"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"   # RTX 4060 = Ada. 없으면 전 아키텍처 빌드로 1시간+
export MAX_JOBS=8                   # nvcc 가 메모리를 많이 먹음. 16코어 전부 쓰면 OOM 위험

# editable 우선 (content/configs 에 M0609 설정을 바로 넣을 수 있어 편하다).
# 그래도 실패하면 일반 설치로 넘어간다.
if ! "$VENV/bin/pip" install -e "$CUROBO_SRC" --no-build-isolation -c "$CONSTRAINTS"; then
    echo "  ! editable 설치 실패 — 일반 설치로 재시도"
    "$VENV/bin/pip" install "$CUROBO_SRC" --no-build-isolation -c "$CONSTRAINTS"
fi

# cuRobo 0.8 은 CUDA 커널을 패키지에 넣지 않는다. 백엔드를 따로 깔아야 한다.
# cuda.core 는 런타임 JIT 이라 컴파일이 필요 없다 (pybind 백엔드는 소스 컴파일).
info "[6b/7] CUDA 커널 백엔드 (cuda.core)"
"$VENV/bin/pip" install -q "cuda-core[cu12]" -c "$CONSTRAINTS"
ok "cuda-core"

# -----------------------------------------------------------------------------
info "[7/7] 최종 검증"
cd "$CUROBO_SRC"
"$VENV/bin/python" - <<'PY' || die "cuRobo 검증 실패"
import time, numpy, torch, curobo
from curobo.kinematics import Kinematics, KinematicsCfg   # 0.8 신 API
from curobo.types import JointState

assert numpy.__version__.startswith("1."), f"numpy 가 {numpy.__version__} 로 밀렸습니다"
assert torch.cuda.is_available(), "torch 가 CUDA 를 못 봅니다"

# GPU 커널이 실제로 도는지 확인 (백엔드 미설치면 여기서 RuntimeError)
kin = Kinematics(KinematicsCfg.from_robot_yaml_file("franka.yml"))
q = torch.zeros(1, len(kin.joint_names), device="cuda")
js = JointState.from_position(q, joint_names=kin.joint_names)
kin.compute_kinematics(js)
torch.cuda.synchronize()

t0 = time.time()
for _ in range(100):
    kin.compute_kinematics(js)
torch.cuda.synchronize()
print(f"  numpy {numpy.__version__} | torch {torch.__version__} | cuRobo {curobo.__version__}")
print(f"  GPU {torch.cuda.get_device_name(0)} | FK {(time.time()-t0)*10:.3f} ms/회")
PY
ok "cuRobo 설치 완료"

python3 -c "import numpy; assert numpy.__version__.startswith('1.')" \
  && ok "시스템 numpy 무사 ($(python3 -c 'import numpy;print(numpy.__version__)'))"

# -----------------------------------------------------------------------------
cat > "$HOME/curobo_env.sh" <<'EOF'
# cuRobo 작업 시작 전에:  source ~/curobo_env.sh
source /opt/ros/humble/setup.bash
source "$HOME/curobo_env/bin/activate"
export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.9"
echo "cuRobo 환경: $(which python)"
EOF

echo
echo "═══════════════════════════════════════════════"
echo " 완료. 앞으로 cuRobo 작업 전에:"
echo "     source ~/curobo_env.sh"
echo "═══════════════════════════════════════════════"
