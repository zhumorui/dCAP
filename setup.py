from setuptools import find_packages, setup

setup(
    name='dcap',
    version='0.1.0',
    description='dCAP open-source release',
    packages=find_packages(include=('dcap', 'dcap.*', 'mmcv', 'mmcv.*')),
    include_package_data=True,
)
