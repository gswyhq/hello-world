import textwrap
from math import floor, ceil

import os

from npre import bbs98
from npre import elliptic_curve as ec

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# 来源： https://github.com/taoxinyi/Proxy-Re-encryption-Demo.git
# cryptography    3.4.7
# npre            0.3  https://github.com/nucypher/nucypher-pre-python  && python3 setup.py install
# 代理重加密解决的问题是：A 在不传递明文的情况下，将加密数据通过CA、proxy传递给B，B再解析出明文的过程。

class CA:
    """
    CA是可以信任的
     生成全局参数
     为用户分发密钥对
    """

    def __init__(self):
        self.pre = bbs98.PRE()
        self.__keypair_dict = {}

    def generate_keypair(self, user):
        sk = self.pre.gen_priv(dtype=bytes)
        pk = self.pre.priv2pub(sk)
        self.__keypair_dict[user.index] = {"pk": pk, "sk": sk}

    def get_public_key(self, user):
        user_index = user.index
        if user_index not in self.__keypair_dict.keys():
            self.generate_keypair(user)
        return self.__keypair_dict[user_index]["pk"]

    def get_secrete_key(self, user):
        user_index = user.index
        if user_index not in self.__keypair_dict.keys():
            self.generate_keypair(user)
        return self.__keypair_dict[user_index]["sk"]

    def get_re_key(self, delegator, delegatee):
        sk_delegator = self.get_secrete_key(delegator)
        sk_delegatee = self.get_secrete_key(delegatee)
        return self.pre.rekey(sk_delegator, sk_delegatee)

    def get_param(self):
        return ec.serialize(self.pre.g)

    def __str__(self):
        return self.__keypair_dict.__str__()


class Proxy():
    """
    半可信代理（服务器）
     在服务器上生成符号密钥（AES）以确保安全
     存储加密的消息
     负责重新加密
    """

    def __init__(self, param):
        self.pre = bbs98.PRE(g=param)
        self.pk_list = []
        self.pre_aes_key = None
        self.pre_seed = None
        self.encrypted_data = None

    def register(self, user, ca):
        if user.index and user.index < len(self.pk_list):
            # This user already registered in Proxy
            return False
        else:
            # Assign index, request to CA Register in pk
            user.index = len(self.pk_list)
            ca.generate_keypair(user)
            self.pk_list.append(ca.get_public_key(user))
            return True

    def aes_encrypt(self, data, pre_seed, user):
        self.pre_seed = pre_seed
        key = os.urandom(32)
        iv = os.urandom(16)  # os.urandom(n), 随即产生n个字节的字符串，可以作为随机加密key使用
        self.pre_aes_key = user.get_encrypted_aes_key(key + iv)
        cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        self.encrypted_data = encryptor.update(data) + encryptor.finalize()
        return self.encrypted_data

    def aes_decrypt(self, aes_encrypted_data, aes_key):
        key = aes_key[0:32]
        iv = aes_key[32:]
        cipher = Cipher(algorithms.AES(key), modes.CTR(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        return decryptor.update(aes_encrypted_data) + decryptor.finalize()

    def pre_reencrypt(self, rekey, pre_encrypted_data):
        return self.pre.reencrypt(rekey, pre_encrypted_data)


class Client():
    """
    客户端
     端对端加密/解密
     为流密码（CHACHA20）生成随机种子
    """

    def __init__(self, param, proxy, ca):
        self.pre = bbs98.PRE(g=param)
        self.index = None
        self.__random_seed = None
        self.__cipher = None
        self.__encryptor = None
        self.__decryptor = None
        # start registry
        proxy.register(self, ca)
        self.__sk = ca.get_secrete_key(self)
        self.__pk = ca.get_public_key(self)

    def generate_random_seed(self):
        key = os.urandom(32)
        nounce = os.urandom(16)
        self.__random_seed = key + nounce

    def get_random_seed(self):
        return self.__random_seed

    def init_chacha20(self, random_seed=None):
        if random_seed:
            self.__random_seed = random_seed
        else:
            self.generate_random_seed()
        algorithm = algorithms.ChaCha20(self.__random_seed[0:32], self.__random_seed[32:])
        self.__cipher = Cipher(algorithm, mode=None, backend=default_backend())
        self.__encryptor = self.__cipher.encryptor()
        self.__decryptor = self.__cipher.decryptor()

    def get_encryption_chacha20(self, data):
        return self.__encryptor.update(data)

    def get_decryption_chacha20(self, encrypted_data):
        return self.__decryptor.update(encrypted_data)

    def get_encrypted_seed(self):
        return self.__pre_encrypt(self.__random_seed)

    def get_decrypted_seed(self, encrypted_seed):
        return self.__pre_decrypt(encrypted_seed)

    def get_decrypted_aes_key(self, encrypted_aes_key):
        return self.__pre_decrypt(encrypted_aes_key)

    def get_encrypted_aes_key(self, aes_key):
        return self.__pre_encrypt(aes_key)

    def __pre_encrypt(self, data):
        return self.pre.encrypt(self.__pk, data)

    def __pre_decrypt(self, data):
        return self.pre.decrypt(self.__sk, data)

    def __str__(self):
        return (b"sk:" + self.__sk + b"pk:" + self.__pk).__str__()

def pretty_print(name, content):
    name = name.ljust(35)
    wrapper = textwrap.TextWrapper(initial_indent=name, width=70,
                                   subsequent_indent=' ' * len(name))
    print(wrapper.fill(content) + '\n')


def print_send_to(left, right, content):
    left = left.ljust(5)
    right = right.rjust(5)
    total_len = len(left) + len(right) + len(content) + 1
    part_len = (70 - total_len) / 2
    print(left + '-' * floor(part_len) + content + '-' * ceil(part_len) + '>' + right + '\n')


def print_send_back(left, right, content):
    left = left.ljust(5)
    right = right.rjust(5)
    total_len = len(left) + len(right) + len(content) + 1
    part_len = (70 - total_len) / 2
    print(left + '<' + '-' * floor(part_len) + content + '-' * ceil(part_len) + right + '\n')


def print_middle(content):
    part_len = (70 - len(content)) / 2
    print('-' * floor(part_len) + content + '-' * ceil(part_len) + '\n')


def main():
    """---------------------------------------初始化------------------------------------------"""
    ca = CA()  # 创建 CA
    param = ca.get_param()  # 生成全局参数（是一个固定值）
    proxy = Proxy(param=param)  # 使用参数创建全局代理（服务器）
    a = Client(param=param, proxy=proxy, ca=ca)  # 使用全局参数、代理、CA创建用户A
    b = Client(param=param, proxy=proxy, ca=ca)  # 使用全局参数、代理、CA创建用户B
    print_send_to('A', 'CA', '请求密钥对')
    print_send_back('A', 'CA', '请求A的密钥对，全局参数')
    print_send_to('B', 'CA', '请求密钥对')
    print_send_back('B', 'CA', '请求B的密钥对，全局参数')

    """---------------------------------------在代理解密A------------------------------------------"""

    print_middle("想在代理上解密一些东西")
    msg = "我是A，我现在分享一些秘密消息哟".encode('utf-8')  # 用户A想给用户B发一些东西
    pretty_print("从A发送的普通消息:", msg.__str__())

    a.init_chacha20()  # 用户 A init chacha20,包含随机种子， 每执行一次初始化，获取的随机种子就变一次；
    seed_a = a.get_random_seed() # init_chacha20初始化后，随机种子就是固定的，但若这期间执行了 generate_random_seed，随机种子就会变动；
    pretty_print("来自A的种子:", seed_a.__str__())

    # ChaCha20是一种流式对称加密算法。
    # 对称加密：加密和解密都是使用的同一个密钥。
    # 非对称加密：加密使用的密钥和解密使用的密钥是不相同的，分别称为：公钥、私钥，公钥和算法都是公开的，私钥是保密的。非对称加密算法性能较低，但是安全性超强，由于其加密特性，非对称加密算法能加密的数据长度也是有限的。
    # 流密码加密过程则是使用初始密钥产生一个伪随机的密码流，连续地处理输入元素序列（通常是用密码流和输入序列做异或运算），产生对应的连续输出序列（密文）。
    # 而流密码解密过程则是采用和加密过程同样的伪随机流产生器算法对初始密钥产生一个相同的伪随机密码流，与密文序列做异或运算，得到明文序列。
    # 分组密码，也叫块加密(block cyphers)，一次加密明文中的一个块。是将明文按一定的位长分组，明文组经过加密运算得到密文组，密文组经过解密运算（加密运算的逆运算），还原成明文组。

    msg_chacha20 = a.get_encryption_chacha20(msg)  # 用户 A 使用 CHACHA20 对 信息进行加密；
    pretty_print("从A 发送CHACHA20 加密的消息:", msg_chacha20.__str__())

    seed_pre_enc_a = a.get_encrypted_seed()  # 用户 A 使用 PRE 和 A的公钥，加密种子
    pretty_print("来自A的PRE加密种子:", seed_pre_enc_a.__str__())
    print_send_to('A', 'PROXY', '(CHACHA20 消息,PRE 加密 种子)')

    aes_enc_msg = proxy.aes_encrypt(msg_chacha20, seed_pre_enc_a, a)  # 代理（服务器）生成随机 AES 密钥并加密 CHACHA20 消息
    pretty_print("代理（服务器）上的AES加密数据:", aes_enc_msg.__str__())
    print_middle("数据在代理（服务器）上解密")

    """---------------------------------------A 从代理（服务器）上下载------------------------------------------"""

    print_middle("用户A当然可以随时查看及下载")
    print_send_to('A', 'PROXY', '请求 CHACHA 20 加密信息')
    print_send_back('A', 'PROXY', '(PRE_enc_seed, PRE_enc_aes_key), 请求普通的 aes_key')

    aes_key_a = a.get_decrypted_aes_key(proxy.pre_aes_key)
    pretty_print("A 重加密 AES 密钥:", aes_key_a.__str__())
    print_send_to('a', 'Proxy', 'plain aes_key')  # A 发送普通的 aes 密钥给代理（服务器）进行解密；

    msg_chacha20_a = proxy.aes_decrypt(proxy.encrypted_data, aes_key_a)  # 代理（服务器）用AES解密消息并发送给用户
    pretty_print("代理解密的 CHACHA20 消息:", msg_chacha20_a.__str__())
    print_send_back('A', 'Proxy', 'CHACHA 20 消息')  # 代理（服务器）解密后发送 CHACHA20 消息给 A

    pre_dec_seed_a = a.get_decrypted_seed(proxy.pre_seed)  # A 从普通种子解密得到 pre_seed
    pretty_print("发送A PRE 解密的种子:", pre_dec_seed_a.__str__())
    a.init_chacha20(pre_dec_seed_a)  # A 通过该种子初始化 CHACHA20
    msg_a = a.get_encryption_chacha20(msg_chacha20_a)
    pretty_print("从A收到普通消息:", msg_a.__str__())
    print_middle("现在 A 成功接收到自己正确的数据了")

    """---------------------------------------A 分享数据给 B------------------------------------------"""
    print_middle("A 想发送数据给 B")

    print_send_to('A', 'CA', '请求B 的 Re-encryption Key')  # A 向CA 请求 B 的 rkey(重加密的key)

    rkey = ca.get_re_key(a, b)  # CA 计算出从A->B重加密密钥（ re-encryption key）并发送给A
    pretty_print("A->B 的重加密密钥（Re-encryption Key） :", rkey.__str__())

    print_send_back('A', 'CA', 'Re-encryption Key A->B')  # CA 计算后发送重加密密钥（re-encryption key）给A
    print_send_to('A', 'Proxy', 'Re-encryption A->B')  # A 发送重加密密钥（ re-encryption key）给代理（服务器）

    pre_re_enc_seed_b = proxy.pre_reencrypt(rkey, seed_pre_enc_a)  # 代理（服务器）给B 计算重加密（re-encrypted） 的种子
    pretty_print("Re-encrypted seed:", pre_re_enc_seed_b.__str__())

    pre_re_enc_aes_key_b = proxy.pre_reencrypt(rkey, proxy.pre_aes_key)  # 代理（服务器）给B 计算重加密（re-encrypted） 的aes密钥（ aes key）
    pretty_print("Re-encrypted AES key:", pre_re_enc_aes_key_b.__str__())

    """---------------------------------------B 从代理(服务器)下载数据------------------------------------------"""

    print_send_back('B', 'Proxy',
                    '(re_enc_seed, re_enc_aes_key), Request plain aes_key')  # 代理(服务器)发送 重加密密钥（re-encryption key）给B;

    aes_key_b = b.get_decrypted_aes_key(pre_re_enc_aes_key_b)  # B 重解密密钥（re-decrypted pre_re_enc_aes_key）, 发送给代理（服务器）
    pretty_print("B 重新解密(re-decrypted)AES密钥 (AES key):", aes_key_b.__str__())
    print_send_to('B', 'Proxy', '普通aes密钥 aes_key')  # B 发送解密后的密钥，即普通的aes key给代理（服务器）

    msg_chacha20_b = proxy.aes_decrypt(proxy.encrypted_data, aes_key_b)  # 代理（服务器）用AES解密消息并返回数据给B
    pretty_print("代理（服务器）解密的CHACHA20 消息:", msg_chacha20_b.__str__())
    print_send_back('B', 'Proxy', 'CHACHA 20 msg')  # 代理（服务器）发送解密后的 CHACHA20 消息给B

    pre_dec_seed_b = b.get_decrypted_seed(pre_re_enc_seed_b)  # B 重新解密pre_re_enc_seed为普通种子
    pretty_print("B重新解密的种子:", pre_dec_seed_b.__str__())
    b.init_chacha20(pre_dec_seed_b)  # B 用解密后的种子初始化 CHACHA20
    msg_b = b.get_encryption_chacha20(msg_chacha20_b)

    pretty_print("B现在收到的普通消息:", msg_b.decode('utf-8'))
    print_middle("现在B成功接收到A分享的数据")

    assert msg == msg_a
    assert msg == msg_b

def test_a2b():
    '''
    # A 分享数据给 B流程（参入的成员有4人：CA、proxy、A、B）：
    # 0、用户 A 使用 PRE 和 A的公钥，生成加密种子 seed_pre_enc_a
    # 1、代理（服务器）生成随机 AES 密钥并加密 A的 CHACHA20 加密消息
    # 2、A 向CA 请求 B 的 rkey(重加密的key)
    # 3、代理（服务器）由 “seed_pre_enc_a+rkey”给B 计算重加密（re-encrypted） 的种子
    # 4、代理（服务器）给B 计算重加密（re-encrypted） 的aes密钥（ aes key）—— pre_re_enc_aes_key_b；
    # 5、B 重解密密钥（re-decrypted pre_re_enc_aes_key）,得到 aes_key_b 发送给代理（服务器）
    # 6、代理（服务器）用 aes_key_b 解密消息并返回数据给B——msg_chacha20_b
    # 7、B解密 msg_chacha20_b得到明文消息
    :return:
    '''
    ca = CA()  # 创建 CA
    param = ca.get_param()  # 生成全局参数（是一个固定值）
    proxy = Proxy(param=param)  # 使用参数创建全局代理（服务器）
    a = Client(param=param, proxy=proxy, ca=ca)  # 使用全局参数、代理、CA创建用户A
    b = Client(param=param, proxy=proxy, ca=ca)  # 使用全局参数、代理、CA创建用户B
    msg = "我是A，我现在分享一些秘密消息哟".encode('utf-8')  # 用户A想给用户B发一些东西
    a.init_chacha20()  # 用户 A init chacha20,包含随机种子（密码）， 每执行一次初始化，获取的随机种子就变一次；
    msg_chacha20 = a.get_encryption_chacha20(msg)  # 用户 A 使用 CHACHA20 对 明文信息进行加密；
    seed_pre_enc_a = a.get_encrypted_seed()  # 用户 A 使用 PRE 和 A的公钥，加密种子，即对密码进行加密；
    aes_enc_msg = proxy.aes_encrypt(msg_chacha20, seed_pre_enc_a, a)  # 代理（服务器）对 A的加密密码+CHACHA20 消息 进行加密；
    rkey = ca.get_re_key(a, b)  # CA 计算出从A->B重加密密钥（ re-encryption key）并发送给A
    pre_re_enc_seed_b = proxy.pre_reencrypt(rkey, seed_pre_enc_a)  # 代理（服务器）给B 计算重加密（re-encrypted） 的种子，及由“A的加密密码及 A->重加密密钥”，计算个新的钥匙；
    pre_re_enc_aes_key_b = proxy.pre_reencrypt(rkey, proxy.pre_aes_key)  # 代理（服务器）给B 计算重加密（re-encrypted） 的aes密钥（ aes key）
    aes_key_b = b.get_decrypted_aes_key(pre_re_enc_aes_key_b)  # B 重解密密钥（re-decrypted pre_re_enc_aes_key）, 发送给代理（服务器）
    msg_chacha20_b = proxy.aes_decrypt(proxy.encrypted_data, aes_key_b)  # 代理（服务器）用AES解密消息并返回数据给B
    pre_dec_seed_b = b.get_decrypted_seed(pre_re_enc_seed_b)  # B 重新解密pre_re_enc_seed为普通种子
    b.init_chacha20(pre_dec_seed_b)  # B 用解密后的种子初始化 CHACHA20
    msg_b = b.get_encryption_chacha20(msg_chacha20_b)
    pretty_print("B现在收到的普通消息:", msg_b.decode('utf-8'))
    assert msg == msg_b

if __name__ == '__main__':
    main()


