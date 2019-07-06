import os
import sys
from setuptools import setup, find_packages

def get_script_path():
  return os.path.dirname(os.path.realpath(sys.argv[0]))

def get_version():
  with open(get_script_path() + '/VERSION', 'r') as f:
    return f.read()

def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
        return requirements
setup(
  name="disturbing-tweets",
  version=get_version(),
  packages=find_packages(),
  install_requires=get_requirements(),
  # package_data={  },
  data_files=[("", ["VERSION"])],  
  author="ktatarnikov",
  author_email="kostyantyn.tatarnikov@gmail.com",
  description="Tweet Classification Pipeline",
  license="Apache2",
  keywords="machine learning tweet classification"
)
