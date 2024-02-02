#!/usr/lib/python3
# -*- coding: utf-8 -*-

# pip3 install fair-esm==2.0.0
# pip3 install "fair-esm[esmfold]"

# OpenFold安装及其他依赖
# pip3 install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
# pip3 install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

# # Colab environment setup
#
# # Install the correct version of Pytorch Geometric.
# import torch
#
# def format_pytorch_version(version):
#   return version.split('+')[0]
#
# TORCH_version = torch.__version__
# TORCH = format_pytorch_version(TORCH_version)
#
# def format_cuda_version(version):
#   return 'cu' + version.replace('.', '')
#
# CUDA_version = torch.version.cuda
# CUDA = format_cuda_version(CUDA_version)
# TORCH, CUDA
# ('1.12.1', 'cu113')
# !pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
# !pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
# !pip install -q torch-cluster -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
# !pip install -q torch-spline-conv -f https://data.pyg.org/whl/torch-{TORCH}+{CUDA}.html
# !pip install -q torch-geometric
#
# # Install esm
# !pip install -q git+https://github.com/facebookresearch/esm.git
#
# # Install biotite
# !pip install -q biotite

# 加载使用预训练模型, 预测蛋白质序列
import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
data = [
    ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein2 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    ("protein3",  "K A <mask> I S Q"),
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)
token_representations = results["representations"][33]

# Generate per-sequence representations via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_representations = []
for i, tokens_len in enumerate(batch_lens):
    sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))

# Look at the unsupervised self-attention map contact predictions
import matplotlib.pyplot as plt
for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    plt.title(seq)
    plt.show()

# 蛋白质结构预测
import torch_geometric
import torch_sparse
from torch_geometric.nn import MessagePassing

import esm
model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
model = model.eval()

# !wget https://files.rcsb.org/download/5YH2.cif -P data/    # save this to the data folder in colab
fpath = 'data/5YH2.cif' # .pdb format is also acceptable
chain_id = 'C'
structure = esm.inverse_folding.util.load_structure(fpath, chain_id)
coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
print('Native sequence:')
print(native_seq)

# !pip install -q py3Dmol
try:
    import py3Dmol

    def view_pdb(fpath, chain_id):
        with open(fpath) as ifile:
            system = "".join([x for x in ifile])

        view = py3Dmol.view(width=600, height=400)
        view.addModelsAsFrames(system)
        view.setStyle({'model': -1, 'chain': chain_id}, {"cartoon": {'color': 'spectrum'}})
        view.zoomTo()
        view.show()

except ImportError:
    def view_pdb(fpath, chain_id):
        print("Install py3Dmol to visualize, or use pymol")

# 展示结构：
view_pdb(fpath, chain_id)

import numpy as np

sampled_seq = model.sample(coords, temperature=1)
print('Sampled sequence:', sampled_seq)

recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
print('Sequence recovery:', recovery)

# Lower sampling temperature typically results in higher sequence recovery but less diversity

sampled_seq = model.sample(coords, temperature=1e-6)
print('Sampled sequence:', sampled_seq)

recovery = np.mean([(a==b) for a, b in zip(native_seq, sampled_seq)])
print('Sequence recovery:', recovery)

ll_fullseq, ll_withcoord = esm.inverse_folding.util.score_sequence(model, alphabet, coords, native_seq)

print(f'average log-likelihood on entire sequence: {ll_fullseq:.2f} (perplexity {np.exp(-ll_fullseq):.2f})')
print(f'average log-likelihood excluding missing coordinates: {ll_withcoord:.2f} (perplexity {np.exp(-ll_withcoord):.2f})')

from copy import deepcopy
masked_coords = deepcopy(coords)
masked_coords[:15] = np.inf # mask the first 10 residues
ll_fullseq, ll_withcoord = esm.inverse_folding.util.score_sequence(model, alphabet, masked_coords, native_seq)

print(f'average log-likelihood on entire sequence: {ll_fullseq:.2f} (perplexity {np.exp(-ll_fullseq):.2f})')
print(f'average log-likelihood excluding missing coordinates: {ll_withcoord:.2f} (perplexity {np.exp(-ll_withcoord):.2f})')

rep = esm.inverse_folding.util.get_encoder_output(model, alphabet, coords)
len(coords), rep.shape

# 资料来源：https://gitcode.com/mirrors/facebookresearch/esm/blob/main/examples/inverse_folding/notebook.ipynb


def main():
    pass


if __name__ == '__main__':
    main()
