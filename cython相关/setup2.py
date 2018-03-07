#! /usr/lib/python3
# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

from pathlib import Path
import shutil
import os

# 使用python setup.py build_ext 构建项目之后
# 用python setup.py bdist_wheel 创建wheel格式！

class MyBuildExt(build_ext):
    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent

        target_dir = build_dir if not self.inplace else root_dir

        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith(".py") and os.sep in root:
                    self.copy_file(Path(os.sep.join(root.split(os.sep)[1:])) / '__init__.py', root_dir, target_dir)
                    self.copy_file(Path(os.sep.join(root.split(os.sep)[1:])) / '__main__.py', root_dir, target_dir)
                    break

        # self.copy_file(Path('auxiliary') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('auxiliary') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('auxiliary/pinyin')/ '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('auxiliary/pinyin')/ '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('cut_word') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('cut_word') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('handlers') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('handlers') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('logger') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('logger') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('model') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('model') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('neo4j') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('neo4j') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('recommend') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('recommend') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('SimilaritySearch') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('SimilaritySearch') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('word_vector') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('word_vector') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('multi_round_dialogue') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('multi_round_dialogue') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('redis_context') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('redis_context') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('conf') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('conf') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('input') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('input') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('mysql_connect') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('mysql_connect') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('oto_qa') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('oto_qa') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('test') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('test') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('contexts') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('contexts') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('dialogue_managem') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('dialogue_managem') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('es_search') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('es_search') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('lucene') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('lucene') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('natural_language_generation') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('natural_language_generation') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('output') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('output') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('third_skill') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('third_skill') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('wmd') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('wmd') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('create_graph') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('create_graph') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('doc') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('doc') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('log') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('log') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('memory') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('memory') / '__main__.py', root_dir, target_dir)
        # self.copy_file(Path('natural_language_understanding') / '__init__.py', root_dir, target_dir)
        # self.copy_file(Path('natural_language_understanding') / '__main__.py', root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        if not (source_dir / path).exists():
            return
        # print('1234323', path, source_dir, destination_dir)
        # print('1232323424', str(source_dir / path), str(destination_dir / path))
        save_file = str(destination_dir / path)
        if not os.path.isdir(os.path.split(save_file)[0]):
            os.makedirs(os.path.split(save_file)[0])
        shutil.copyfile(str(source_dir / path), save_file)

module_list = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith(".py") and os.sep not in root:
            assert '__init__.py' not in files, "根目录不应该有`__init__.py`文件"
            module = Extension(os.path.splitext(file)[0], [file])
            module_list.append(module)
            if file == 'setup.py':
                continue
        elif file.endswith(".py") and os.sep in root:
            assert '__init__.py' in files, "路径`{}`,缺失`__init__.py`文件".format(root)
            module = Extension('.'.join(root.split(os.sep)[1:] + ['*']), [os.sep.join(root.split(os.sep)[1:] + ['*.py'])])
            module_list.append(module)
            break

setup(
    name="mypkg",

    ext_modules=cythonize(
        module_list,
        # [
        #     Extension("tornado_server", ["tornado_server.py"]),
        #     Extension("tornado_parse", ["tornado_parse.py"]),
        #     Extension("auxiliary.*", ["auxiliary/*.py"]),
        #     Extension("auxiliary.pinyin.*", ["auxiliary/pinyin/*.py"]),
        #     Extension("cut_word.*", ["cut_word/*.py"]),
        #     Extension("handlers.*", ["handlers/*.py"]),
        #     Extension("logger.*", ["logger/*.py"]),
        #     Extension("model.*", ["model/*.py"]),
        #     Extension("neo4j.*", ["neo4j/*.py"]),
        #     Extension("recommend.*", ["recommend/*.py"]),
        #     Extension("SimilaritySearch.*", ["SimilaritySearch/*.py"]),
        #     Extension("word_vector.*", ["word_vector/*.py"]),
        #     Extension("multi_round_dialogue.*", ["multi_round_dialogue/*.py"]),
        #     Extension("redis_context.*", ["redis_context/*.py"]),
        #     Extension("conf.*", ["conf/*.py"]),
        #     Extension("input.*", ["input/*.py"]),
        #     Extension("mysql_connect.*", ["mysql_connect/*.py"]),
        #     Extension("oto_qa.*", ["oto_qa/*.py"]),
        #     Extension("test.*", ["test/*.py"]),
        #     Extension("contexts.*", ["contexts/*.py"]),
        #     Extension("dialogue_managem.*", ["dialogue_managem/*.py"]),
        #     Extension("es_search.*", ["es_search/*.py"]),
        #     Extension("lucene.*", ["lucene/*.py"]),
        #     Extension("natural_language_generation.*", ["natural_language_generation/*.py"]),
        #     Extension("output.*", ["output/*.py"]),
        #     Extension("third_skill.*", ["third_skill/*.py"]),
        #     Extension("wmd.*", ["wmd/*.py"]),
        #     Extension("create_graph.*", ["create_graph/*.py"]),
        #     Extension("doc.*", ["doc/*.py"]),
        #     Extension("log.*", ["log/*.py"]),
        #     Extension("memory.*", ["memory/*.py"]),
        #     Extension("natural_language_understanding.*", ["natural_language_understanding/*.py"]),
        #
        # ],
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

# python3 setup.py build_ext

