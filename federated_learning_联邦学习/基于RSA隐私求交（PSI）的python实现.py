#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
隐私保护下的样本ID匹配
数据对齐
数据对齐的难点在于如何让隐私信息不被暴露的情况下对齐数据，也就是说比如A有{u1,u2,u3,u4}四个数据，B有{u1,u2,u3,u5}四个数据，
如果让A,B在不知道对方的数据前提下，找到{u1,u2,u3}这个交集。
FATE SecureBoost是使用了Gang Liang and Sudarshan S. Chawathe的《Privacy-Preserving Inter-Database Operations》论文中的方法。
总体上的思路就是：
1.B先通过RSA生成n（公有密钥）、e（加密算法）、d（解密算法），然后把公钥(n,e)传给A。
2.A拿到公钥后，先将自己的数据通过哈希函数进行加密，然后通过增加噪声r,然后通过加密算法(e)加密然后回传给B。
3.B拿到加密后的Y(A)后，使用解密算法(d)解密，这时获得含有噪声的哈希值Z(A),然后B同时将自己的数据进行哈希然后在进行解密，再进行二次哈希，传给A。
4.A拿到后，先消除Z(A)的噪声，然后再进行哈希，得到D(A)，这样D(A)和Z(B)都是两方数据进行二次哈希和解密后的值，也就是处于同一纬度。这时候就可以求两者的交集。然后回传给B.
5.B拿到后通过比对自己数据和交集后的数据，得到实际的交集{u1,u2,u3}.然后回传给A.

通过上面的操作，A,B在不知道对方的差分数据下得到了交集，也就是对齐了数据样本。

'''

# pip install pycryptodome

import Crypto.Cipher as Cipher
from Crypto.Cipher import PKCS1_OAEP
import Crypto.Signature as Sign
import Crypto.Hash as Hash
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_v1_5_cipper
from Crypto.Signature import PKCS1_v1_5 as PKCS1_v1_5_sign
from Crypto.Hash import SHA1
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256

# 读取标准的rsa公私钥pem文件
def load_rsa_file(fn):
    key = None
    try:
        key = RSA.importKey(open(fn).read())
    except Exception as err:
        print('导入rsa的KEY文件出错', fn, err)
    return key

# 标准字符串密钥转rsa格式密钥
def rsa_key_str2std(skey):
    ret = None
    try:
        ret = RSA.importKey(skey)
    except Exception as err:
        print('字符串密钥转rsa格式密钥错误', skey, err)
    return ret


# RSA_加密
def rsa_enc(data, rsa_key):
    ciphertext = b''
    try:
        cipher = PKCS1_OAEP.new(rsa_key)
        ciphertext = cipher.encrypt(data)
    except Exception as err:
        print('RSA加密失败', '', err)
    return ciphertext

'''
由于RSA在加密过程中，每次加密只能加密最大长度的字符串，如果你的加密数据超长，在加密过程中需要分段加密，同理，解密也是分段解密的。
1024位的证书，加密时最大支持117个字节，解密时为128； 2048位的证书，加密时最大支持245个字节，解密时为256。
加密时支持的最大字节数：证书位数/8 -11（比如：2048位的证书，支持的最大加密字节数：2048/8 – 11 = 245） 其中，11位字节为保留字节。 
若密钥文件是2048比特，所以加密分块长度为245字节。
'''

# RSA在解密分段时与加密时用的分段大小无关，都是按照密钥长度/8来分段解密的。
# RSA解密
def rsa_dec(data, rsa_key):
    ret_data = b''
    try:
        cipher = PKCS1_OAEP.new(rsa_key)
        ret_data = cipher.decrypt(data)
    except Exception as err:
        print('RSA解密失败', '', err)
    return ret_data

# RSA签名
def rsa_sign(data, rsa_key):
    signature = ''
    try:
        h = SHA256.new(data)
        signature = pkcs1_15.new(rsa_key).sign(h)
    except Exception as err:
        print('RSA签名失败', '', err)
    return signature

# RSA签名验证
def rsa_sign_verify(data, sig, rsa_key):
    try:
        h = SHA256.new(data)
        pkcs1_15.new(rsa_key).verify(h, sig)
        ret = True
    except (ValueError, TypeError):
        ret = False
    return ret

class Rsa:
    """RSA加解密签名类
    """
    def __int__(self, ciper_lib=PKCS1_v1_5_cipper, sign_lib=PKCS1_v1_5_sign, hash_lib=SHA1,
                pub_file=None, pri_file=None, pub_skey=None, pri_skey=None, pub_key=None, pri_key=None,
                reversed_size=11):

        # 加解密库
        self.ciper_lib = ciper_lib
        self.sign_lib = sign_lib
        self.hash_lib = hash_lib

        # 公钥密钥
        if pub_key:
            self.pub_key = pub_key
        elif pub_skey:
            self.pub_key = RSA.importKey(pub_skey)
        elif pub_file:
            self.pub_key = RSA.importKey(open(pub_file).read())

        if pri_key:
            self.pri_key = pri_key
        elif pri_skey:
            self.pri_key = RSA.importKey(pri_skey)
        elif pri_file:
            self.pri_key = RSA.importKey(open(pri_file).read())

        # 分块保留长度
        self.block_reversed_size = reversed_size

    # 根据key长度计算分块大小
    def get_block_size(self, rsa_key):
        try:
            # RSA仅支持限定长度内的数据的加解密，需要分块
            # 分块大小
            reserve_size = self.block_reversed_size
            key_size = rsa_key.size_in_bits()
            if (key_size % 8) != 0:
                raise RuntimeError('RSA 密钥长度非法')

            # 密钥用来解密，解密不需要预留长度
            if rsa_key.has_private():
                reserve_size = 0

            bs = int(key_size / 8) - reserve_size
        except Exception as err:
            print('计算加解密数据块大小出错', rsa_key, err)
        return bs

    # 返回块数据
    def block_data(self, data, rsa_key):
        bs = self.get_block_size(rsa_key)
        for i in range(0, len(data), bs):
            yield data[i:i + bs]

    # 加密
    def enc_bytes(self, data, key=None):
        text = b''
        try:
            rsa_key = self.pub_key
            if key:
                rsa_key = key

            cipher = self.ciper_lib.new(rsa_key)
            for dat in self.block_data(data, rsa_key):
                cur_text = cipher.encrypt(dat)
                text += cur_text
        except Exception as err:
            print('RSA加密失败', data, err)
        return text

    # 解密
    def dec_bytes(self, data, key=None):
        text = b''
        try:
            rsa_key = self.pri_key
            if key:
                rsa_key = key

            cipher = self.ciper_lib.new(rsa_key)
            for dat in self.block_data(data, rsa_key):
                if type(self.ciper_lib) == Cipher.PKCS1_OAEP:
                    cur_text = cipher.decrypt(dat)
                else:
                    cur_text = cipher.decrypt(dat, '解密异常')
                text += cur_text
        except Exception as err:
            print('RSA解密失败', data, err)
        return text

    # RSA签名
    def sign_bytes(self, data, key=None):
        signature = ''
        try:
            rsa_key = self.pri_key
            if key:
                rsa_key = key

            h = self.hash_lib.new(data)
            signature = self.sign_lib.new(rsa_key).sign(h)
        except Exception as err:
            print('RSA签名失败', '', err)
        return signature

    # RSA签名验证
    def sign_verify(self, data, sig, key=None):
        try:
            rsa_key = self.pub_key
            if key:
                rsa_key = key
            h = self.hash_lib.new(data)
            self.sign_lib.new(rsa_key).verify(h, sig)
            ret = True
        except (ValueError, TypeError):
            ret = False
        return ret


from Crypto.PublicKey import RSA
import hashlib
import binascii
import gmpy2  # 为了理解中间“盲化”和哈希的过程，所以没有用现成的RSA加解密方案 而是用gmpy2 库的加速模幂计算方法做了 加解密操作。
import os

rand_bits = 128


def hash_bignumber(num, method='sha1'):
    '''
        num: an integer
    '''
    if method == 'sha1':
        hash_obj = hashlib.sha1(str(num).encode('utf-8'))
        digest_hex = hash_obj.hexdigest()
        return int(digest_hex, 16)


def gen_key():
    """
    生成公钥、私钥
    """
    key = RSA.generate(1024)
    pk = (key.n, key.e)  # 公钥
    sk = (key.n, key.d) # 私钥
    return pk, sk


def blind_msg_arr_use_pk(msg_arr, pk):
    """
    先将自己的数据通过哈希函数进行加密，然后通过增加噪声,然后通过加密算法(e)加密然后回传给B。
    msg_arr: 待加密的数据
    pk: 公钥
    """
    msg_hash_number_blind = []
    rand_private_list = []
    for item in msg_arr:
        hash_num = hash_bignumber(item)
        hash_num = hash_num % pk[0]

        ra = int(binascii.hexlify(os.urandom(rand_bits)), 16)
        cipher_ra = gmpy2.powmod(ra, pk[1], pk[0])
        rand_private_list.append(ra)
        msg_hash_number_blind.append(hash_num * cipher_ra)

    return msg_hash_number_blind, rand_private_list


def deblind_arr_from_client(hash_arr_blind, sk):
    deblind_hash_arr = []

    for item in hash_arr_blind:
        de_blind_number = gmpy2.powmod(item, sk[1], sk[0])
        deblind_hash_arr.append(de_blind_number)
    return deblind_hash_arr


def enc_and_hash_serverlist(server_list, sk):
    hash_server_list = []
    for item in server_list:
        hash_num = hash_bignumber(item)
        c_hash_num = gmpy2.powmod(hash_num, sk[1], sk[0])
        hash_server_list.append(hash_bignumber(c_hash_num))
    return hash_server_list


def hash_deblind_client_arr(deblind_hash_arr, rand_list, pk):
    db_client = []
    for item, ra in zip(deblind_hash_arr, rand_list):
        ra_inv = gmpy2.invert(ra, pk[0])  # ra*ra_inv == 1 mod n
        db_client.append(hash_bignumber((item * ra_inv) % pk[0]))
    return db_client


def get_common_elements_idx(db_client, db_server):
    #在O（n^2）复杂性中搜索交集元素
    #返回本地数据列表中的交集元素索引
    common_set_index = []
    for idx in range(len(db_client)):
        rec_a = db_client[idx]
        for rec_b in db_server:
            if rec_a == rec_b:
                common_set_index.append(idx)
    return common_set_index



# B: 服务端；A: 客户端

# 第一步：B先通过RSA生成n（公有密钥）、e（加密算法）、d（解密算法），然后把公钥(n,e)传给A。
# 服务器端：
# #RSA密钥生成并向客户端发送pk
pk, sk = gen_key()

# 第二步：A拿到公钥后，先将自己的数据通过哈希函数进行加密，然后通过增加噪声r,然后通过加密算法(e)加密然后回传给B。
# 客户端遮蔽(blind)本地数据，并发送给服务端
msg_arr_client = [12, 3, 4, 8, 10, 23]
blind_arr, rlist = blind_msg_arr_use_pk(msg_arr_client, pk)

# 第三步：B拿到加密后的Y(A)后，使用解密算法(d)解密，这时获得含有噪声的哈希值Z(A),然后B同时将自己的数据进行哈希然后在进行解密，再进行二次哈希，传给A。
# 服务器端从客户端接收 blind 数据并用密钥将其解解密
received_blind_arr = blind_arr.copy()
deblind_hash_arr = deblind_arr_from_client(received_blind_arr, sk)

server_list = [12, 3, 4, 5, 1, 32, 45]  # 12,3,4,8,10,23
# 加密(encrypt)并哈希服务端的数据
hashed_server_list = enc_and_hash_serverlist(server_list, sk)
# 服务端将 deblind 数据 和 哈希 数据，发送给客户端

# 第四步：A拿到后，先消除Z(A)的噪声，然后再进行哈希，得到D(A)，这样D(A)和Z(B)都是两方数据进行二次哈希和解密后的值，也就是处于同一纬度。这时候就可以求两者的交集。然后回传给B.
# 客户端
# 接收 deblind 数据 和 服务端的 哈希 数据
received_deblind_hash_arr = deblind_hash_arr.copy()
db_server = hashed_server_list.copy()

# 对 dblind 数据进行哈希
db_client = hash_deblind_client_arr(received_deblind_hash_arr, rlist, pk)

# 第五步：B拿到后通过比对自己数据和交集后的数据，得到实际的交集{u1,u2,u3}.然后回传给A.
common_index_local_list = get_common_elements_idx(db_client, db_server)

# db_client
# Out[26]:
# [518057341372717878674147115022451794502814328633,
#  339817230836041501254319482928426404832612759526,
#  43618899339596131425792965685688348931510658906,
#  672100316914535805431569431702314817521696668920,
#  543087380102566481978566419476126491456527741046,
#  444834945103673130598713078794734857650604151418]
# db_server
# Out[27]:
# [518057341372717878674147115022451794502814328633,
#  339817230836041501254319482928426404832612759526,
#  43618899339596131425792965685688348931510658906,
#  95262380750846508252989948307858470767648121078,
#  1357233762374240953219328150941627664320073514114,
#  340773062468855542802207839428439565755220533834,
#  1221475363234810991605908444253382213053833058680]

print('客户端的哈希数据:', db_client)
print('服务端的哈希数据:', db_server)

for idx, true_id in zip(common_index_local_list, [0, 1, 2]):
    assert idx == true_id

# common_index_local_list
# Out[29]: [0, 1, 2]
def main():
    pass


if __name__ == '__main__':
    main()
