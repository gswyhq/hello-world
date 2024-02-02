#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# RDKit是用 C++ 和 Python 编写的化学信息学和机器学习软件的集合。在本教程中，RDKit 用于方便高效地将 SMILES转换为分子对象，然后从中获得原子和键的集合。
#
# SMILES 以 ASCII 字符串的形式表示给定分子的结构。
# SMILES 字符串是一种紧凑的编码，对于较小的分子而言，它相对易于人类阅读。
# 将分子编码为字符串既减轻并促进了给定分子的数据库和/或网络搜索。
# RDKit 使用算法将给定的 SMILES 准确地转换为分子对象，然后可用于计算大量分子属性/特征。

# pip -q install rdkit

import os

USERNAME = os.getenv("USERNAME")

# rdkit安装成功了，但导入rdkit包报错：ImportError: DLL load failed while importing rdBase: 找不到指定的模块。
# 解决方法：
from ctypes import WinDLL
libs_dir = os.path.abspath(fr'D:\Users\{USERNAME}\AppData\Roaming\Python\Python39\site-packages\rdkit.libs')
with open(os.path.join(libs_dir, '.load-order-rdkit-2022.3.5')) as file:
    load_order = file.read().split()
for lib in load_order:
    WinDLL(os.path.join(libs_dir, lib))

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage
from rdkit import Chem
from rdkit.Chem import Draw, AllChem

# 定义一个SMILES字符串
smi = 'CCCc1nn(C)c2C(=O)NC(=Nc12)c3cc(ccc3OCC)S(=O)(=O)N4CCN(C)CC4'
m = Chem.MolFromSmiles(smi)
# m = Chem.MolFromMol2File(fr"D:\Users\{USERNAME}\Downloads\img\out.mol2")  # 读取mol2文件
# Draw.ShowMol(m, size=(150, 150), kekulize=False)  # 窗口展示图形
# Draw.MolToFile(m, 'data/output.png', size=(150, 150)) # 保存图形到文件；
img = Draw.MolToImage(m, size=(1500, 1500))
img.save(fr"D:\Users\{USERNAME}\Downloads\img\mol.jpg")

# 3D展示
m3d = Chem.MolFromSmiles('CNC(=O)N(N(CCCl)S(C)(=O)=O)S(C)(=O)=O')
m3d = Chem.AddHs(m3d)  # 为了得到靠谱的三维构象，一般先加氢
AllChem.EmbedMolecule(m3d, randomSeed=3)  # 通过距离几何算法计算3D坐标
AllChem.MMFFOptimizeMolecule(m3d)  # 转换完后再进行一步力场优化，比如MMFF94
Draw.ShowMol(m3d, size=(250,250))

# SMILES（Simplified molecular input line entry system），简化分子线性输入规范，是一种用ASCII字符串明确描述分子结构的规范。
# 由于SMILES用一串字符来描述一个三维化学结构，它必然要将化学结构转化成一个生成树，此系统采用纵向优先遍历树算法。
# 转化时，先要去掉氢，还要把环打开。表示时，被拆掉的键端的原子要用数字标记，支链写在小括号里。
# SMILES字符串可以被大多数分子编辑软件导入并转换成二维图形或分子的三维模型。转换成二维图形可以使用Helson的“结构图生成算法”（Structure Diagram Generation algorithms）。

# 典范SMILES保证每个化学分子只有一个SMILES表达式。典范SMILES常用于分子数据库的索引。
# 记法：
# 1．原子用在方括号内的化学元素符号表示。
# 例如[Au]表示“金”，氢氧根离子是[OH-]。
# 有机物中的C、N、O、P、S、Br、Cl、I等原子可以省略方括号，其他元素必须包括在方括号之内。
# 2．氢原子常被省略。
# 对于省略了方括号的原子，用氢原子补足价数。
# 例如，水的SMILES就是O，乙醇是CCO。
# 3．双键用“=”表示；三键用“#”表示。
# 含有双键的二氧化碳则表示为O=C=O，含有三键的氰化氢表示为C#N。
# 4．如果结构中有环，则要打开。断开处的两个原子用同一个数字标记，表示原子间有键相连。
# 环己烷(C6H12)表示为C1CCCCC1。需要注意，标志应该是数字(在此例中为1)而不是“C1”这个组合。扩展的表示是(C1)-(C)-(C)-(C)-(C)-(C)-1而不是(C1)-(C)-(C)-(C)-(C)-(C)-(C1)。
# 5．芳环中的C、O、S、N原子分别用小写字母c,o,s,n表示。
# 6．碳链上的分支用圆括号表示。
# 比如丙酸表示为CCC(=O)O，FC(F)F或者C(F)(F)F表示三氟甲烷。
# 7. 在芳香结构中的N原子上连有一个Ｈ原子，用[nH]表示
# 8. 用@和@@表示手性
# 异构SMILES
# 异构SMILES是指扩展的，可以表示同位素、手性和双键结构的SMILES版本。它的一个显著特征是可以精确地说明局部手性。
# 双键两侧的结构分别用符号/和\表示，例如，F/C=C/F表示反二氟乙烯，它的两个氟原子位于双键的两侧。
# 而F/C=C\F表示顺二氟乙烯，它的两个氟原子位于双键的同一侧。

# 生成三维图像
# 主要特点是：
# 可以从SMILES字符串中读取分子，并保留立体化学信息。
# 在没有初始三维坐标的情况下，这些分子和任何其他分子都可以生成三维构象。
# 含有立体化学信息的SMILES字符串可以从分子中生成。
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
m = Chem.MolFromSmiles('N[C@@]([H])(CC(=O)O)C(=O)N[C@]([H])(CCCCN)C(=O)O')
m3d=Chem.AddHs(m)  # 在rdkit中，分子在默认情况下是不显示氢的，但氢原子对于真实的几何构象计算有很大的影响，所以在计算3D构象前，需要使用Chem.AddHs()方法加上氢原子。
AllChem.EmbedMolecule(m3d, randomSeed=1)  # 生成3D构象
img = Draw.MolToImage(m3d, size=(250,250))
img.save(fr"D:\Users\{USERNAME}\Downloads\img\mol3d.jpg")  # 但结果看起来跟二维差不多；

###########################################################################################################################
# openbabel支持SMILES，PDB，MOL2，SDF等等110种化学分子格式文件，其中绝大部分都是可以相互转换的，例如1维的SMILES转化为2维或者3维的MOL2等等，加全氢原子，或者极性氢原子，分子能量最小化处理。
# 同时可以计算分子指纹，比较分子相似度，例如：MACCS，ECFP系列，FCFP系列，FP系列
# 此外还可以计算一系列的2D，3D分子描述符。
from openbabel import pybel
import os
USERNAME = os.getenv('USERNAME')
mol = pybel.readstring("smi", "N[C@@]([H])(CC(=O)O)C(=O)N[C@]([H])(CCCCN)C(=O)O")
# mol.make3D(forcefield="mmff94", steps=500)
mol.localopt()
mol.write(format="mol2", filename=fr"D:\Users\{USERNAME}\Downloads\img\mol3d.mol2", overwrite=True)
mol.write(format="_png2", filename=fr"D:\Users\{USERNAME}\Downloads\img\mol3d.png", overwrite=True, opt={"p": 1500})  # 参数p，控制图片清晰度，image size, default 300


# 使用 OpenBabel 示例可以参考:
# https://open-babel.readthedocs.io/en/latest/UseTheLibrary/PythonDoc.html
from openbabel import pybel
mol = pybel.readstring("smi", "C1=CC=CS1")
mol.addh()
mol.make3D()
mol.write("mol2", "out.mol2")

mol = next(pybel.readfile('mol2', fr"D:\Users\{USERNAME}\Downloads\img\out.mol2"))
mol.make3D()
mol.write(format="_png2", filename=fr"D:\Users\{USERNAME}\Downloads\img\mol3d.png", overwrite=True, opt={"p": 5000})  # 参数p，控制图片清晰度，image size, default 300

# 实际上在命令行执行的话，就简单多了；
# obabel -:"C1=CC=CS1" --gen3D -O out.mol2

# 也可以通过如下方式执行
# import os
# os.system("obabel -:\"C1=CC=CS1\" --gen3D -O out.mol2")

# SMILES字符串转换为三维立体图像，也可以参考：https://www.ccdc.cam.ac.uk/Community/blog/SMILES-to-3D-chemical-structure-CSD/

def main():
    pass


if __name__ == '__main__':
    main()
