from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.12.0','tensorflow_probability']

setup(
  name='trainer',
  version='0.1',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  description='Generic example trainer package with dependencies.')
