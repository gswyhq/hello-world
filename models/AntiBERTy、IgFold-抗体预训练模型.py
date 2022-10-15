#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip install antiberty==0.0.5
from antiberty import AntiBERTy, get_weights
antiberty = AntiBERTy.from_pretrained(get_weights())

########################################################################################################################
# 使用 IgFold预测之前，需要安装：
# PyRosetta openmm pdbfixer bioconda anarci

# openmm是用于分子模拟的高性能工具包。
# 重新编号
# 抗体重新编号默认使用ANARCI。

# 注意：首次初始化IgFoldRunner时，它将下载预先训练的权重。这可能需要几分钟时间，并且需要网络连接。
# 从序列预测抗体结构
# 成对抗体序列可以作为序列字典提供，其中键是链名称，值是序列。

from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # 输出 PDB file
    sequences=sequences, # 抗体序列，Antibody sequences
    do_refine=True, # 使用PyRosetta优化抗体结构
    do_renum=True, # 重新编号预测抗体结构 (Chothia)
)

########################################################################################################################
# 要预测 纳米抗体(nanobody)结构（或单个重链或轻链），只需提供一个序列：
from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS"
}
pred_pdb = "my_nanobody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, #  纳米抗体序列 Nanobody sequence
    do_refine=True, # 抗体结构使用PyRosetta优化
    do_renum=True, # 重新编号预测抗体结构 (Chothia)
)

########################################################################################################################
# 若要预测结果，而不是优化(refinement),需设置 do_refine=False:

from igfold import IgFoldRunner

sequences = {
    "H": "QVQLQESGGGLVQAGGSLTLSCAVSGLTFSNYAMGWFRQAPGKEREFVAAITWDGGNTYYTDSVKGRFTISRDNAKNTVFLQMNSLKPEDTAVYYCAAKLLGSSRYELALAGYDYWGQGTQVTVS"
}
pred_pdb = "my_nanobody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Nanobody sequence
    do_refine=False, # Refine the antibody structure with PyRosetta
    do_renum=True, # Renumber predicted antibody structure (Chothia)
)

########################################################################################################################
# 抗体结构预测 RMSD
# RMSD 估值，在输出的PDB文件的B因子序列中，这些值也是共fold方法中获取；

from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
out = igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Antibody sequences
    do_refine=True, # Refine the antibody structure with PyRosetta
    do_renum=True, # Renumber predicted antibody structure (Chothia)
)

out.prmsd # 预测 RMSD 通过每个残基的 N, CA, C, CB 原子 (dim: 1, L, 4)

########################################################################################################################
# 抗体序列嵌入 Antibody sequence embedding

from igfold import IgFoldRunner

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}

igfold = IgFoldRunner()
emb = igfold.embed(
    sequences=sequences, # Antibody sequences
)

emb.bert_embs # AntiBERTy最终隐藏层的嵌入 (dim: 1, L, 512)
emb.gt_embs # Embeddings after graph transformer layers (dim: 1, L, 64)
emb.strucutre_embs # Embeddings after template incorporation IPA (dim: 1, L, 64)

########################################################################################################################
# 通过设置use_penmm=True，可以将OpenMM优化优先于PyRosetta。
from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Antibody sequences
    do_refine=True, # Refine the antibody structure with PyRosetta
    use_openmm=True, # 使用 OpenMM 优化
    do_renum=True, # Renumber predicted antibody structure (Chothia)
)

########################################################################################################################
# 通过设置 use_abnum =True，可以将使用AbNum服务重新编号优先于ANARCI。

from igfold import IgFoldRunner, init_pyrosetta

init_pyrosetta()

sequences = {
    "H": "EVQLVQSGPEVKKPGTSVKVSCKASGFTFMSSAVQWVRQARGQRLEWIGWIVIGSGNTNYAQKFQERVTITRDMSTSTAYMELSSLRSEDTAVYYCAAPYCSSISCNDGFDIWGQGTMVTVS",
    "L": "DVVMTQTPFSLPVSLGDQASISCRSSQSLVHSNGNTYLHWYLQKPGQSPKLLIYKVSNRFSGVPDRFSGSGSGTDFTLKISRVEAEDLGVYFCSQSTHVPYTFGGGTKLEIK"
}
pred_pdb = "my_antibody.pdb"

igfold = IgFoldRunner()
igfold.fold(
    pred_pdb, # Output PDB file
    sequences=sequences, # Antibody sequences
    do_refine=True, # Refine the antibody structure with PyRosetta
    do_renum=True, # Renumber predicted antibody structure (Chothia)
    use_abnum=True, # Send predicted structure to AbNum server for Chothia renumbering
)



def main():
    pass


if __name__ == '__main__':
    main()
