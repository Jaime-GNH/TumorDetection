from setuptools import setup, find_packages

setup(
    name='RL_TumorDetection',
    version='0.0.0.0',
    packages=find_packages(
        where='RL_TD'
    ),
    include_package_data=True,
    install_requires=[
        'torch==2.1.0',
        'opencv-python==4.8.1.78',
        'pip==23.2.1',
        'requests',
        'importlib-metadata; python_version == "3.10.9"',
    ],
)
