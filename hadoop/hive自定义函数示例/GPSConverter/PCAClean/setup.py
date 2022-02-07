from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = [ 'ProvinceCityAreaClean.py', 'main.c']

extensions = [Extension("ProvinceCityAreaClean", sourcefiles, 
  include_dirs=['/root/java/jdk1.8.0_191/include/',
    '/root/java/jdk1.8.0_191/include/linux/',
    '/usr/local/include/python3.6m/'],
  library_dirs=['/usr/local/lib/'],
  libraries=['python3.6m'])]

setup(ext_modules=cythonize(extensions, language_level = 3))

