#!/usr/bin/env python

import os
import sys
from setuptools import setup

version = '0.0.2'

if sys.argv[-1] == 'publish':
    os.system("python3 setup.py sdist upload")
    os.system("python3 setup.py bdist_wheel upload")
    print("You probably want to also tag the version now:")
    print("  git tag -a %s -m 'version %s'" % (version, version))
    print("  git push --tags")
    sys.exit()

if sys.argv[-1] == 'tag':
    os.system("git tag -a %s -m 'version %s'" % (version, version))
    os.system("git push --tags")
    sys.exit()

setup(
    name='semantics_match',  # pip list,查询到的名字
    version=version,
    author='gswyhq',
    author_email='gswyhqyang@web1.co',
    url='http://192.168.3.101/gswyhq/semantics_match',
    description='语义匹配，先用bm25初步过滤，再结合wmd匹配的分数，给出最终的结果',
    long_description=open('README.md').read(),

    # https://pypi.python.org/pypi?:action=list_classifiers
    classifiers=[
        'Development Status :: 1 - Planning',  # 开发状态
        'Intended Audience :: Developers',  # 计划读者群
        'License :: OSI Approved :: BSD License',  # 版权许可
        'Operating System :: POSIX :: Linux',  # 操作系统
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Python Modules',  # 主题
    ],
    py_modules=['semantics_match'],  # import semantics_match，导入时候的包名

    # 告诉Distutils需要处理那些包（包含__init__.py的文件夹）
    packages=['semantics_match', 'semantics_match.main', 'semantics_match.auxiliary', 'semantics_match.conf',
              'semantics_match.lucene', 'semantics_match.wmd', 'semantics_match.logger'],

    # 通常包含与包实现相关的一些数据文件或类似于readme的文件。如果没有提供模板，会被添加到MANIFEST文件中。
    # package_data = {'': ['*.txt'], 'mypkg': ['data/*.dat'],} 表示包含所有目录下的txt文件和mypkg/data目录下的所有dat文件。
    package_data={'semantics_match': ['model/wx_vector_char.pkl',
                                      'data/kefu_weibo_doc_count_freq.json',
                                      'data/defined_word_antonym.txt',
                                     'data/user_word_synonym.txt',
                                     'data/word_antonym.txt',
                                     'data/word_synonym.txt'
                                    ]},
    include_package_data = True,
    install_requires=['jieba>=0.38',
                    'numpy==1.13.0',
                    'scikit-learn==0.18.1',
                    'gensim==2.0.0',
                    'pyemd==0.4.3'
                    ],
    test_suite='tests',  # 运行“python3 setup.py test” 即可测试当前目录下的tests目录中的单元测试文件
)