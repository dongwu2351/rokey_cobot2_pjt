
# 1. The topic(/dsr01/msg/current_posx) was removed by Doosan github at 2025.12.10
# 
# self.create_subscription(
#       Float64MultiArray, "/dsr01/msg/current_posx", self.current_posx_callback, 10)
#
# 
# 2. The topic(/dsr01/msg/joint_state) was changed to (/dsr01/joint_states)by Doosan github at 2025.12.12
# 
#   self.create_subscription(
#     Float64MultiArray, "/dsr01/msg/joint_state", self.joint_state_callback, 10)
#
# so the both topic above were implemented by service
# the /dsr01/joint_states was available both service and topic
#

import rclpy
from rclpy.node import Node

import tkinter as tk
from tkinter import StringVar
import threading

import time # add
import DR_init # add

from dsr_msgs2.srv import SetRobotMode

# Configuration for a single robot
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY, ACC = 60, 60
ON, OFF = 1, 0

# Initialize DR_init with robot parameters
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL


def copy_to_clipboard(root, text_box):
    text = text_box.get().strip()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()  # 클립보드 업데이트


def create_entries(root, default_value, row, col):
    entry_var = StringVar()
    entry_var.set(str(round(default_value, 3)))
    entry = tk.Entry(root, textvariable=entry_var, width=50)
    entry.grid(row=row, column=col, padx=10, pady=5)
    return entry_var


class ServiceClinetNode(Node):
    def __init__(self):
        super().__init__("service_client_node")
        # 수정: create_client의 f"" 부분 수정
        self.cli = self.create_client(SetRobotMode, f"/{ROBOT_ID}/system/set_robot_mode")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            print("Waiting for service...")
            pass

    def send_request(self, mode=0):
        request = SetRobotMode.Request()
        request.robot_mode = mode
        future = self.cli.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def ros_thread(text_var1, text_var2):
    # DSR_ROBOT2 함수 import
    try:
        from DSR_ROBOT2 import get_current_posx, get_current_posj
    except ImportError as e:
        print(f"Error importing DSR_ROBOT2: {e}")
        return

    while rclpy.ok():
        try:
            # posx 서비스 호출
            posx_res = get_current_posx()
            if posx_res is not None:
                data_x = [round(d, 3) for d in posx_res[0]]
                text_var1.set(f"posx({data_x})")

            # posj 서비스 호출
            posj_res = get_current_posj()
            if posj_res is not None:
                data_j = [round(d, 3) for d in posj_res]
                text_var2.set(f"posj({data_j})")

            time.sleep(0.1)

        except Exception:
            time.sleep(0.5)
            continue

    rclpy.shutdown()


def main():
    # Tkinter GUI 실행
    root = tk.Tk()
    tk.Label(root, text="current_posx:").grid(row=0, column=0)
    text_var1 = create_entries(root, 0.0, 0, 1)
    tk.Button(root, text="copy", command=lambda: copy_to_clipboard(root, text_var1)).grid(
        row=0, column=3, padx=2, pady=5
    )

    tk.Label(root, text="joint_state:").grid(row=1, column=0)
    text_var2 = create_entries(root, 0.0, 1, 1)
    tk.Button(root, text="copy", command=lambda: copy_to_clipboard(root, text_var2)).grid(
        row=1, column=3, padx=2, pady=5
    )

    # 서비스 실행
    print("Service Start")
    rclpy.init()

    # [추가함] 전역 노드 설정
    dsr_node = rclpy.create_node("dsr_global_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    client_node = ServiceClinetNode()
    response = client_node.send_request(0)
    client_node.get_logger().info(f"results: {response}")

    # ROS2 스레드 실행
    ros = threading.Thread(target=ros_thread, args=(text_var1, text_var2))
    ros.start()

    root.mainloop()


if __name__ == "__main__":
    main()