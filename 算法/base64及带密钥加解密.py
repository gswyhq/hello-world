#!/usr/bin/env python
# coding=utf-8

import base64
import gradio as gr
import hashlib
# pip3 install pycryptodome
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def encode(input_str):
    """base64加密"""
    return base64.b64encode(input_str.encode('utf-8')).decode('utf-8')

def decode(input_str):
    """base64解密"""
    encoded_bytes = input_str.encode('utf-8')
    return base64.decodebytes(encoded_bytes).decode('utf-8')

def encodeV2(input_str, key_str=None):
    """
    带秘钥的加密
    :param input_str: 待加密字符串
    :param key: 秘钥
    :return:
    """
    if not key_str:
        key = get_random_bytes(16)  # 生成随机的16字节密钥
        key_hex = key.hex()
    else:
        key_hex = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:32]
        key = bytes.fromhex(key_hex)

    plaintext = input_str.encode('utf-8') # b'Hello, world!'  # 加密数据
    cipher = AES.new(key, AES.MODE_ECB)  # 创建 AES 加密器对象
    padded_plaintext = pad(plaintext, AES.block_size)  # 填充明文数据
    ciphertext = cipher.encrypt(padded_plaintext)  # 加密
    return ciphertext.hex(), key_hex

def decodeV2(input_str, key_hex):
    """
    带秘钥的解密
    :param input_str:
    :param key:
    :return:
    """
    ciphertext = bytes.fromhex(input_str)
    key = bytes.fromhex(key_hex)
    cipher = AES.new(key, AES.MODE_ECB)  # 创建 AES 加密器对象
    decrypted = cipher.decrypt(ciphertext)  # 解密
    decrypted_data = unpad(decrypted, AES.block_size)  # 去除填充
    return decrypted_data.decode('utf-8')

with gr.Blocks() as demo:
    gr.Markdown("《emo互助协会🏸🚁⛏》群专用文本加解密系统！")
    with gr.Tabs():
        with gr.TabItem("文本加密(无秘钥)"):
            text_input = gr.Textbox(placeholder="今天天气怎么样？", label="输入待加密的文本")
            text_output = gr.Textbox(placeholder="5LuK5aSp5aSp5rCU5oCO5LmI5qC377yf", label="文本加密结果")
            text_button = gr.Button("加密")
        with gr.TabItem("文本解密(无秘钥)"):
            with gr.Row():
                text_input2 = gr.Textbox(placeholder="5aSp5rCU5LiN6ZSZIQ==", label="输入待解密的文本")
                text_output2 = gr.Textbox(placeholder="天气不错!", label="文本解密结果")
            image_button = gr.Button("解密")

        with gr.TabItem("文本加密(带秘钥)"):
            text_input3 = gr.Textbox(placeholder="今天天气怎么样？", label="输入待加密的文本")
            text_key = gr.Textbox(placeholder="李白", label="输入秘钥口令")
            text_button2 = gr.Button("加密", elem_id="custom-button")
            text_output3 = gr.Textbox(placeholder="be69669b33cb78fe21b0d4e95ce4fe73136ac51aed1f090fa991ab6b882e3ccd", label="文本加密结果")
            text_key_hex = gr.Textbox(placeholder="b23f840ff52e01233100b176e00b332e", label="解密秘钥")


        with gr.TabItem("文本解密(带秘钥)"):
            text_input4 = gr.Textbox(placeholder="be69669b33cb78fe21b0d4e95ce4fe73136ac51aed1f090fa991ab6b882e3ccd", label="输入待解密的文本")
            text_key_hex4 = gr.Textbox(placeholder="b23f840ff52e01233100b176e00b332e", label="解密秘钥")
            image_button2 = gr.Button("解密", elem_id="custom-button")
            text_output4 = gr.Textbox(placeholder="今天天气怎么样？", label="文本解密结果")

    text_button.click(fn=encode, inputs=text_input, outputs=text_output)
    image_button.click(fn=decode, inputs=text_input2, outputs=text_output2)

    text_button2.click(fn=encodeV2, inputs=[text_input3, text_key], outputs=[text_output3, text_key_hex])
    image_button2.click(fn=decodeV2, inputs=[text_input4, text_key_hex4], outputs=text_output4)

# 自定义 CSS, 设置按钮颜色等
demo.css = """
    #custom-button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        padding: 10px 24px;
    }
"""

def main():
    demo.launch(server_name='0.0.0.0',share=True,
                # auth=('admin', '666'),
                auth_message="欢迎登录文本加解密系统", debug=True)


if __name__ == "__main__":
    main()




