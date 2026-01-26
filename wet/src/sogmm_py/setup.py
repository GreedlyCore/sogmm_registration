from setuptools import setup, find_packages

setup(
   name='sogmm_py',
   packages=find_packages(where='src'),
   package_dir={'':'src'}
)
