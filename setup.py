import os
import sys
import pkg_resources

from setuptools import setup, find_packages

os.environ['CC'] = 'g++'

setup(name='FireFLy',
      version='0.0.1',
      packages=find_packages(include=['FireFly']),
      include_package_data=True,
)