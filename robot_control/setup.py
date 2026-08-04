from setuptools import find_packages, setup
from glob import glob
import os
from pathlib import Path

package_name = 'robot_control'
setup_dir = Path(__file__).resolve().parent
workspace_dir = (
    setup_dir.parent.parent
    if setup_dir.parent.name == 'build'
    else setup_dir.parent
)


def first_existing(*paths):
    for path in paths:
        if os.path.isfile(path):
            return path
    return None


resource_files = glob('resource/*')
calibration_path = first_existing(
    'resource/T_gripper2camera.npy',
    os.path.relpath(
        workspace_dir
        / 'corecode'
        / 'Calibration_Tutorial'
        / 'T_gripper2camera.npy',
        setup_dir,
    ),
    os.path.relpath(
        workspace_dir
        / 'pick_and_place_text'
        / 'resource'
        / 'T_gripper2camera.npy',
        setup_dir,
    ),
)
if calibration_path and calibration_path not in resource_files:
    resource_files.append(calibration_path)

setup(
    name=package_name,
    version='0.0.0',
    packages=['cobot_robot_control'],
    package_dir={'cobot_robot_control': 'robot_control'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), resource_files),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_control = cobot_robot_control.robot_control:main',
        ],
    },
)
