#!/usr/bin/env python
# coding=utf-8

# 第一步：下载转换代码 git clone https://github.com/1079863482/paddle2torch_PPOCRv3
# 第二步：下载PaddleOCR模型；
# 1、cd paddle2torch_PPOCRv3/weights/
# 2、wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar
# 3、wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
# 4、tar -xvf ch_PP-OCRv3_rec_train.tar && tar -xvf ch_PP-OCRv3_det_distill_train.tar
# 第三步：模型转换
# 1、cd paddle2torch_PPOCRv3
# 2、python3 paddle2torch_ppocrv3_det.py
# 3、python3 paddle2torch_ppocrv3_rec.py

import os
import sys
import numpy as np
import torch
import onnxruntime

USERNAME = os.getenv("USERNAME")
sys.path.append(rf"D:\Users\{USERNAME}\github_project/paddle2torch_PPOCRv3")
det_model = torch.load(rf"D:\Users\{USERNAME}\github_project/paddle2torch_PPOCRv3/weights\ppv3_det.pt")
rec_model = torch.load(rf"D:\Users\{USERNAME}\github_project/paddle2torch_PPOCRv3/weights\ppv3_rec.pt")


data_arr1 = torch.ones(1,3,640,640)
torch_infer = det_model(data_arr1).detach().numpy()
print("det:", torch_infer)

data_arr2 = torch.ones(1, 3, 48, 224)
torch_infer2 = rec_model(data_arr2).detach().numpy()
print("rec:", torch_infer2)

################################################################################################################################################

det_onnx_path = rf"D:\Users\{USERNAME}\github_project/paddle2torch_PPOCRv3/weights\ppv3_db.onnx"
rec_onnx_path = rf"D:\Users\{USERNAME}\github_project/paddle2torch_PPOCRv3/weights\ppv3_rec.onnx"
det_onnx_model = onnxruntime.InferenceSession(det_onnx_path)
np_arr1 = np.array(data_arr1).astype(np.float32)

onnx_infer = det_onnx_model.run(None, {'input': np_arr1})
print("det onnx:", onnx_infer[0])

rec_onnx_model = onnxruntime.InferenceSession(rec_onnx_path)
np_arr2 = np.array(data_arr2).astype(np.float32)

onnx_infer = rec_onnx_model.run(None, {'input': np_arr2})
print("rec onnx:", onnx_infer[0])

def main():
    pass


if __name__ == "__main__":
    main()

