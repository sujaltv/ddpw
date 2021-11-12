import os
from setuptools import setup, find_packages

install_requires = []
req_path = os.path.dirname(os.path.realpath(__file__)) + '/requirements.txt'
with open(req_path) as f:
  install_requires = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
  long_description = f.read()

with open("version.txt", "r", encoding="utf-8") as f:
  version = f.read()

setup(
  name="ddpw",
  version=version,
  author="Sujal T.V.",
  url="http://ddpw.projects-tvs.surge.sh",
  description=r"""A utility package to encapsulate the scaffolding for PyTorch's
  Distributed Data Parallel""",
  long_description=long_description,
  long_description_content_type="text/markdown",
  packages=find_packages(),
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: MIT License",
      "Operating System :: Mac :: Linux :: Windows",
  ],
  python_requires='>=3.8',
  install_requires=install_requires
)