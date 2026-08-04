from setuptools import find_packages, setup
from glob import glob
import os
from pathlib import Path

package_name = 'object_detection'
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
model_path = first_existing(
    'resource/yolov8n_tools_0122.pt',
    os.path.relpath(
        workspace_dir
        / 'pick_and_place_text'
        / 'resource'
        / 'yolov8n_tools_0122.pt',
        setup_dir,
    ),
)
if model_path and model_path not in resource_files:
    resource_files.append(model_path)

fruit_model_path = first_existing(
    'resource/fruits_best.pt',
    os.path.relpath(
        workspace_dir / 'fruits' / 'best.pt',
        setup_dir,
    ),
)
if fruit_model_path and fruit_model_path not in resource_files:
    resource_files.append(fruit_model_path)

setup(
    name=package_name,
    version='0.0.0',
    packages=['cobot_object_detection'],
    package_dir={'cobot_object_detection': 'object_detection'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'resource'), resource_files),
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
            'object_detection = cobot_object_detection.detection:main',
        ],
    },
)
