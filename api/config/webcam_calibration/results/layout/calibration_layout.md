# Prompt 04 calibration layout

Operator faces the robot workspace from the front.

```text
                    rear-left upper
                         CAM2
                           \
                            \
       CAM0  -------- robot/workspace -------- CAM1
 front-center upper                         front-right upper
                            /
                 WRIST_DEPTH on gripper/wrist
```

- CAM0: operator front-center upper, USB root path `1-5`, frame `cam0_optical`.
- CAM1: operator front-right upper, USB root path `1-1`, frame `cam1_optical`.
- CAM2: operator rear-left upper, USB root path `1-8`, frame `cam2_optical`.
- WRIST_DEPTH: D435I serial `207122078284`, USB 3 path `2-2`, frame `wrist_depth_optical`.
- SPARE_DEPTH: disconnected and not required.
- Fixed-camera board capture: measured ChArUco handle clamped by RG2 and hand-guided.
- Wrist calibration/holdout: board fixed on the worktable; robot hand-guided around it.
- Camera mounts, robot base and wrist mount were not moved during the corresponding calibration runs.
