#!/usr/bin/env python

import os
import sys
from setuptools import setup

version = "0.3"

if sys.argv[-1] == 'publish':
    os.system("python3 setup.py sdist upload")
    os.system("python3 setup.py bdist_wheel upload")
    print("You probably want to also tag the version now:")
    print("  git tag -a %s -m 'version %s'" % (version, version))
    print("  git push --tags")
    sys.exit()

if sys.argv[-1] == 'tag':
    os.system("git tag -a %s -m 'version %s'" % (version, version))
    # os.system("git push --tags")
    os.system("git push origin tag %s"%version)
    sys.exit()
    
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
        'Development Status :: 4 - Beta',  # 开发状态
        'Intended Audience :: Developers',  # 计划读者群
        'License :: OSI Approved :: BSD License',  # 版权许可
        'Operating System :: OS Independent',  # 操作系统
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',  
        'Topic :: Software Development :: Libraries :: Python Modules',  # 主题
    ],
    py_modules=['pytrie'],
    install_requires=['sortedcontainers'],
    test_suite='tests',  # 运行“python3 setup.py test” 即可测试当前目录下的tests目录中的单元测试文件
)
