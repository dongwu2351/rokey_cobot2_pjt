from setuptools import find_packages, setup
import glob
import os

package_name = 'pick_and_place_voice'


def first_existing(*paths):
    for path in paths:
        if os.path.isfile(path):
            return path
    return None


resource_files = glob.glob('resource/*') + glob.glob('resource/.*')
for asset in (
    first_existing(
        'resource/yolov8n_tools_0122.pt',
        '../pick_and_place_text/resource/yolov8n_tools_0122.pt',
    ),
    first_existing(
        'resource/T_gripper2camera.npy',
        '../corecode/Calibration_Tutorial/T_gripper2camera.npy',
        '../pick_and_place_text/resource/T_gripper2camera.npy',
    ),
):
    if asset and asset not in resource_files:
        resource_files.append(asset)


setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=find_packages(include=[
        'robot_control', 
        'voice_processing', 
        'object_detection'
    ]),

    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/resource', resource_files),
        ('share/' + package_name + '/launch', glob.glob('launch/*.launch.py')),
        # ('share/ament_index/resource_index/packages',['resource/' + 'voice_processing']),
        # ('share/voice_processing', ['package.xml']),
        # ('share/object_detection', ['package.xml']),
        # ('share/robot_control', ['package.xml']),

        # ('share/' + package_name + '/launch', glob.glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey4090',
    maintainer_email='rokey4090@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'robot_control = robot_control.robot_control:main',
            'webcam_pick_place = robot_control.webcam_pick_place:main',
            'chat_commander = robot_control.chat_commander:main',
            'object_detection = object_detection.detection:main',
            'yolo_view = object_detection.yolo_view:main',
            'get_keyword = voice_processing.get_keyword:main',
        ],
    },
)
