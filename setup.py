# Copyright 2021 John Kitchin
# (see accompanying license files for details).
from setuptools import setup

setup(name='f21_06623',
      version='0.0.1',
      description='support for F21 06-623',
      url='http://github.com/jkitchin/f21-06623',
      maintainer='John Kitchin',
      maintainer_email='jkitchin@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['pycse'],
      setup_requires=[],
      data_files=['requirements.txt', 'LICENSE'],
      install_requires=[],
      long_description='''Just support functions for multiple choice questions.''')

# (shell-command "python setup.py register") to setup user
# to push to pypi - (shell-command "python setup.py sdist upload")


# Set TWINE_USERNAME and TWINE_PASSWORD in .bashrc
# python setup.py sdist bdist_wheel
# twine upload dist/*
