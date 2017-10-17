#!/usr/bin/env python

from setuptools import setup

setup(
    name='PyTrie',
    version='0.3',
    author='George Sakkis',
    author_email='george.sakkis@gmail.com',
    url='https://github.com/gsakkis/pytrie/',
    description='A pure Python implementation of the trie data structure.',
    long_description=open('README.md').read(),

    # https://pypi.python.org/pypi?:action=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    py_modules=['pytrie'],
    install_requires=['sortedcontainers'],
    test_suite='tests',  # 运行“python3 setup.py test” 即可测试当前目录下的tests目录中的单元测试文件
)