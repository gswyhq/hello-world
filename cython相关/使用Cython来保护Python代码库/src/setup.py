#! /usr/lib/python3
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

from pathlib import Path
import shutil

# 使用python setup.py build_ext构建项目之后
# 用python setup.py bdist_wheel创建wheel格式！

class MyBuildExt(build_ext):
    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent

        target_dir = build_dir if not self.inplace else root_dir

        self.copy_file(Path('mypkg') / '__init__.py', root_dir, target_dir)
        self.copy_file(Path('mypkg2') / '__init__.py', root_dir, target_dir)
        self.copy_file(Path('mypkg') / '__main__.py', root_dir, target_dir)
        self.copy_file(Path('mypkg2') / '__main__.py', root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        if not (source_dir / path).exists():
            return

        shutil.copyfile(str(source_dir / path), str(destination_dir / path))



setup(
    name="mypkg",
    ext_modules=cythonize(
        [
           Extension("mypkg.*", ["mypkg/*.py"]),
           Extension("mypkg2.*", ["mypkg2/*.py"])
        ],
        build_dir="build",
        compiler_directives=dict(
            # always_allow_keywords指令通过禁用具有大量参数的函数只允许使用关键字参数这一优化，
            # 使Flask视图函数可以正常工作。
            always_allow_keywords=True
        )),
    # cmdclass=dict(
    #     build_ext=build_ext
    # ),
    # 可以在构建项目的其余部分后从源代码树中复制__init__.py文件。 一个很好的方法是覆盖setup.py中的build_ext类：
    cmdclass=dict(
        build_ext=MyBuildExt
    ),
    # packages=["mypkg", "mypkg2"]

    # 在调用setup时删除packages参数中的包名。这样，仍然可以构建扩展并包含在wheel中，但源代码将不会在其中。
    packages=[]
)

