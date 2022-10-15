#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
from transformers import ViTFeatureExtractor, TFViTForImageClassification
# from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests

USERNAME = os.getenv("USERNAME")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(rf"D:\Users\{USERNAME}\Pictures\000000039769.jpg")

# https://huggingface.co/apple/mobilevit-small

# 预训练模型来源：https://huggingface.co/facebook/convnext-tiny-224
feature_extractor = ConvNextFeatureExtractor.from_pretrained(rf"D:\Users\{USERNAME}\data\convnext-tiny-224")
model = ConvNextForImageClassification.from_pretrained(rf"D:\Users\{USERNAME}\data\convnext-tiny-224", from_tf=True)


inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

def main():
    pass


if __name__ == '__main__':
    main()
