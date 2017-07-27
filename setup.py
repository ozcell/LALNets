from setuptools import setup
from setuptools import find_packages


setup(name='lalnets',
      version='0.1.3',
      description='Ozsel Kilinc LALNets',
      author='Ozsel Kilinc',
      author_email='ozselkilinc@gmail.com',
      install_requires = ['keras==1.2.1',
                          'sklearn', 'pandas', 'numpy', 'scipy', 'matplotlib'],
      packages=find_packages())
