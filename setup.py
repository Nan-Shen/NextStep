#!/usr/bin/env python

from __future__ import division

from setuptools import setup
from glob import glob
import ast
import re

__author__ = "Nan Shen"
__copyright__ = ""
__credits__ = ["Nan Shen"]
__license__ = ""
__maintainer__ = "Nan Shen"
__email__ = "nanshenbms@gmail.com"

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('NextStep/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(name='NextStep',
      version=version,
      description='Stock price prediction',
      classifiers=[
          'Development Status :: 1 - Alpha',
          'License :: ',
          'Programming Language :: Python :: 2.7',
          'Topic :: ',
      ],
      url='https://github.com/Nan-Shen/NextStep.git',
      author=__author__,
      author_email=__email__,
      license=__license__,
      packages=['lodi'],
      scripts=glob('scripts/*py'),
      install_requires=[
          'numpy',
          'scipy',
          'click',
          'seaborn',
          'sklearn',
          'pandas',
          'statsmodels'
      ],
      zip_safe=False)
