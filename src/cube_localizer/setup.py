import os
from glob import glob
from setuptools import setup

package_name = 'cube_localizer'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), ['config.rviz']),
    ],
    install_requires=['setuptools', 'tf_transformations'],
    zip_safe=True,
    maintainer='JG',
    maintainer_email='Jerry.Gabrie@gmail.com',
    description='Cube localization package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cube_localizer = cube_localizer.cube_localizer:main',
        ],
    },
)