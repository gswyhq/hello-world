#!/usr/bin/python3
# coding: utf-8

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

"""
compile command 'python compile.py build_ext --inplace'
"""

file_list = ['gaodun_product.py',
 'ocr_product.py',
 'GaoDun_rank_api.py',
 'data_clean.py',
 'fuzzymatch.py',
 'elastic_search.py',
 'kwd.py',
 'kwe.py',
 'wmd.py',
 'utils.py',
 'data_object.py',
 'similarity_search.py',
 'data_transformation.py']


ext_modules = [
#    Extension("gaodun_product",  ["gaodun_product.py"]),
#    Extension("mymodule2",  ["mymodule2.py"]),
#   ... all your modules that need be compiled ...
Extension(file.split('.')[0],  [file]) for file in file_list
]
print(ext_modules)
setup(
    name = 'My Program Name',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)

def main():
    pass


if __name__ == '__main__':
    main()