# rokey_cobot2_pjt — visual-servo tracking & snatching on a Doosan M0609

A wrist-mounted RealSense + YOLO tracks a tool in real time. The arm hovers
over it and follows it as it moves; pressing **G** makes the arm intercept it
on a diagonal and grab it — including while the object is still moving.

The tracking controller is one file:
[`pick_and_place_voice/robot_control/robot_control.py`](pick_and_place_voice/robot_control/robot_control.py).

---

## It is not one folder

Three separate things have to be running. Only the last one lives in this repo:

| Piece | Where it comes from |
|---|---|
| RealSense camera driver | apt: `ros-humble-realsense2-camera` |
| Doosan M0609 driver (`dsr_bringup2`) | **separate workspace** `~/cobot_ws`, cloned from [ROKEY-SPARK/doosan-robot2_2026](https://github.com/ROKEY-SPARK/doosan-robot2_2026) @ `a8fdcdc` |
| Tracking / snatching controller | **this repo**, `pick_and_place_voice` |

`~/cobot_ws` is the *underlay*, this workspace is the *overlay*. Source them in
that order or `dsr_msgs2` won't be found.

---

## Setup on a new machine

Needs Ubuntu 22.04 + ROS 2 Humble already installed.

```bash
git clone https://github.com/yoon-taehwan/rokey_cobot2_pjt.git ~/cobot2_ws
cd ~/cobot2_ws
bash scripts/setup_workspace.sh
```

That clones and builds the Doosan driver into `~/cobot_ws`, installs the pinned
Python packages, and builds this workspace. Re-running it is safe.

### Two pins that are load-bearing

- **`pymodbus==2.5.3`** — `onrobot.py` imports `pymodbus.client.sync`, removed
  in 3.x. On 3.x the gripper silently fails to connect and the arm moves
  without gripping.
- **`vcs-versioning`** — colcon's `ament_python` build imports `setuptools_scm`,
  which needs it. Without it *every* Python package fails to build with
  `ModuleNotFoundError: No module named 'vcs_versioning'`.

---

## Running

Every new terminal needs both workspaces, **driver first**:

```bash
source ~/cobot_ws/install/setup.bash
source ~/cobot2_ws/install/setup.bash
```

**Terminal 1** — camera + robot driver:

```bash
ros2 launch pick_and_place_voice bringup.launch.py
# add rviz:=true for RViz, host:=<ip> if the controller is elsewhere
```

**Terminal 2** — the controller (opens an OpenCV window; it needs focus for keys):

```bash
ros2 run pick_and_place_voice robot_control
```

### Keys

| Key | Action |
|---|---|
| `F` | Lock onto the highest-scoring detection and start following |
| `G` | Snatch it — diagonal intercept and grab |
| **`SPACE`** | **Stop now.** Aborts a descent immediately |
| `P` | Toggle motion prediction (lead + feedforward) |
| `D` | Toggle dry-run |
| `H` | Home |
| `O` / `C` | Gripper open / close |
| `ESC` | Quit (stops motion first) |

**It starts in dry-run.** Nothing moves until you press `D`. The overlay shows
`target vel: est N -> used N mm/s`: if `est` is large but `used` is 0, that is
the noise gate working — a motionless object should read `used 0`.

### Voice (optional)

Keyboard tracking needs no API key. For the "Hello Rokey → name a tool" path,
put a real key in `pick_and_place_voice/resource/.env` (see `.env.example`) and
also run:

```bash
ros2 run pick_and_place_voice get_keyword
```

---

## Per-cell calibration — read before trusting another robot

Two things in this repo describe **one specific physical setup** and are wrong
anywhere else:

1. **`pick_and_place_voice/resource/T_gripper2camera.npy`** — the hand-eye
   transform for this camera on this bracket. A different mount (or a knock
   hard enough to shift it) invalidates it, and the arm will confidently reach
   for the wrong place.

   Redo it with the scripts in `corecode/Calibration_Tutorial/`:

   ```bash
   cd corecode/Calibration_Tutorial
   python3 data_recording.py        # jog the arm, capture poses + images -> data/
   python3 handeye_calibration.py   # solve -> T_gripper2camera.npy
   python3 verify.py                # sanity-check the result
   ```

   **That writes `T_gripper2camera.npy` next to itself, which is not the copy
   the robot reads.** Copy it across and rebuild, or the arm keeps using the
   old calibration:

   ```bash
   cp T_gripper2camera.npy ../../pick_and_place_voice/resource/
   cd ../.. && colcon build --packages-select pick_and_place_voice
   ```

2. **`FOLLOW_X/Y/Z_MIN/MAX` in `robot_control.py`** — the safe-workspace box.
   These are hand-picked placeholders, *not* derived from the robot's reachable
   volume. The controller refuses to descend on a target outside the box, so
   values that are too tight look like "it won't grab", and values that are too
   loose let it drive somewhere it shouldn't.

Network defaults, also cell-specific: robot `192.168.1.100:12345`, gripper
(Modbus) `192.168.1.1:502`, PC on `192.168.1.x`.

---

## Layout

```
pick_and_place_voice/          <- the tracking package
  robot_control/
    robot_control.py           <- tracking, interception, state machine, GUI
    onrobot.py                 <- RG2 gripper over Modbus
  object_detection/
    realsense.py               <- camera topic subscriber
    yolo.py                    <- YOLO wrapper
    yolo_view.py               <- optional ByteTrack debug view
    detection.py               <- older service node (unused by tracking)
  voice_processing/            <- wake word, STT, GPT keyword extraction
  resource/                    <- YOLO weights, hand-eye calibration, .env
  launch/bringup.launch.py     <- camera + robot driver

od_msg/                        <- service type used only by detection.py
pick_and_place_text/           <- earlier terminal-input version
object_detection/, robot_control/, voice_processing/   (top level)
                               <- earlier 3-package split; code is duplicated
                                  and object_detection/ was later reused for a
                                  fruit-detection exercise
corecode/                      <- course material, incl. hand-eye calibration
dum_E_project/                 <- design docs
fruits/                        <- separate 221MB fruit-detection experiment
                                  (gitignored)
```

## Not committed

`.gitignore` keeps out `build/ install/ log/`, `__pycache__/`, the 221MB
`fruits/` experiment, and **`.env`** — which holds a real OpenAI key. If you
ever need to share the key, do it out of band, not through this repo.
