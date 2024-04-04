from setuptools import setup, find_packages

setup(
    name='TumorDetection',
    version='0.0.0.0',
    packages=find_packages(
        where='TumorDetection'
    ),
    include_package_data=True,
    install_requires=[
        'pip==23.2.1',
        'torch==2.1.1',
        'torchvision==0.16.1',
        'torchmetrics==1.2.0',
        'torchinfo==1.8.0',
        'lightning==2.1.0',
        'scikit-learn==1.3.1',
        'opencv-python==4.8.1.78',
        'numpy==1.26.0',
        'requests',
        'importlib-metadata; python_version == "3.10.9"',
    ],
)
