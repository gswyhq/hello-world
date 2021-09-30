#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 同态加密是一种对称加密算法，由Craig Gentry发明提出。其同态加密方案包括4个算法，即密钥生成算法、加密算法、解密算法和额外的评估算法。
# 全同态加密包括两种基本的同态类型，即乘法同态和加法同态，加密算法分别对乘法和加法具备同态特性。
# 全同态加密保证了数据处理方无法知道所处理的数据的明文信息，可以直接对数据的密文进行相应的处理，这样以来，用户的信息资料可以得到相应的安全保障。
# 比如 B想分析A的数据，但A又不想把明文数据给B，这个时候可以采用同态加密，其过程：
# 1， A 将加密的数据给到B；
# 2，B对A给的加密数据进行分析（仅限于有限的几种）
# 3, B把分析结果发送给A，A将分析结果解密，给到B。
# 需要注意的是，上面示例中，明文与分析结果间，不能有简单的相关性，否则，B很容易根据分析结果推算出原明文。

from phe import paillier
import numpy as np

import gmpy2 as gy
# pip3 install gmpy2 安装失败的话，可以下载whl文件进行安装，如：
# pip install Downloads/gmpy2-2.0.8-cp36-cp36m-win_amd64.whl
# whl文件下载地址：https://www.lfd.uci.edu/~gohlke/pythonlibs/#gmpy
import random
import time
import libnum

def test_1():
    # 生成公钥和私钥：
    public_key, private_key = paillier.generate_paillier_keypair()

    secret_number_list = [3.141592653, 300, -4.6e-12]

    # 采用公钥对数据进行加密：
    encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]

    # 采用私钥，对数据进行解密
    [private_key.decrypt(x) for x in encrypted_number_list]

    a1, b1, c1 = secret_number_list
    a, b, c = encrypted_number_list

    # 加密数据，可以与明文数据加减乘除，加密数据可以与加密数据进行加减；

    a_plus_5 = a + 5
    a_plus_b = a + b
    a_times_3_5 = a * 3.5
    a_minus_1_3 = a - 1             # = a + (-1)
    a_div_minus_3_1 = a / -3.1      # = a * (-1 / 3.1)
    a_minus_b = a - b               # = a + (b * -1)


    enc_mean = np.mean(encrypted_number_list)
    enc_dot = np.dot(encrypted_number_list, [2, -400.1, 5318008])

    # 但加密数据间不能进行乘除；加密数据不能作为被除数；
    # a * b
    # NotImplementedError: Good luck with that...
    #
    # 1 / a
    # TypeError: unsupported operand type(s) for /: 'int' and 'EncryptedNumber'

    a_times_3_5_lp = a * paillier.EncodedNumber.encode(a.public_key, 3.5, 1e-2)

    # 钥匙环
    keyring = paillier.PaillierPrivateKeyring()
    keyring.add(private_key)
    public_key1, private_key1 = paillier.generate_paillier_keypair(keyring)
    public_key2, private_key2 = paillier.generate_paillier_keypair(keyring)

    [private_key.decrypt(x) for x in encrypted_number_list]
    # [3.141592653, 300, -4.6e-12]

    [keyring.decrypt(x) for x in encrypted_number_list]
    # [3.141592653, 300, -4.6e-12]

    # A的私钥不能解密B的加密数据；
    try:
        [private_key1.decrypt(x) for x in [public_key2.encrypt(x) for x in secret_number_list]]
    except Exception as e:
        print('A的私钥不能解密B的加密数据', e)

    # 但钥匙环可以解密密钥匙环上任何一把私钥对应公钥加密的数据；
    [keyring.decrypt(x) for x in encrypted_number_list]

    # 私钥解密 密文计算的结果，并验证是否与明文计算的一致：
    print(keyring.decrypt(sum(encrypted_number_list)) , sum(secret_number_list))
    print(keyring.decrypt(sum(encrypted_number_list)) == sum(secret_number_list))  # 因为精度的问题，可能有点不一致；

    print(keyring.decrypt(sum(encrypted_number_list[:2])) , sum(secret_number_list[:2]))
    print(keyring.decrypt(sum(encrypted_number_list[:2])) == sum(secret_number_list[:2]))

    print(keyring.decrypt(a*5+2.2) , a1*5+2.2)


class Paillier(object):
    '''
    在进行加解密之前，必须先产生可以用来加密的公钥n和g。n是两个大小相近的两个大素数的乘积：n=p·q。
    g是Zn2中的半随机数，同时g的阶必须在Z∗n2中并且能被n整除。
    由于g必须符合一些特殊性质（我们将在解密部分提出）所以Z∗n2中会有很少一部分元素不能用作g，意味着g是一个半随机数。
    为了简单计算，我们先选取两个小素数p=7，q=11计算得到n=p·q=77。从Z∗n2中选择g（g的阶必须是Z∗n2中元素并且是n的倍数。
    除此之外，g需要满足的另一个性质将会在解密时详细描述），在这里我们先选择5652作为g。
    因为g模n2的阶是2310且是77的倍数，并且在Z∗n2中。
    那么g所需要的包括未清楚定义的所有性质将会被满足。
    至此，我们找到了用来实际加解密运算过程的公钥（n，g）。随着公钥发布，任何人都能使用公钥加密数据并将密文传给私钥持有者。
    '''
    def __init__(self, pubKey=None, priKey=None):
        self.pubKey = pubKey
        self.priKey = priKey

    def __gen_prime__(self, rs):
        """
        生成素数，注意，当传递信息越长，生成的素数需要更大
        :param rs:
        :return:
        """
        p = gy.mpz_urandomb(rs, 1024)
        while not gy.is_prime(p):
            p += 1
        return p

    def __L__(self, x, n):
        res = gy.div((x - 1), n)
        # this step is essential, directly using "/" causes bugs
        # due to the floating representation in python
        return res

    def __key_gen__(self):
        '''
        首先选择两个大素数 $p$ 和 $q$，计算出 $n$ 为 $p$ 和 $q$ 的乘积。并取一个随机数 $g$（通常取 $n+1$）。$n$ 和 $g$ 作为公钥。
        然后根据卡迈克尔函数计算私钥 $\lambda$ 为 $p-1$ 和 $q-1$ 的乘积。
        :return:
        '''
        # generate random state
        while True:
            rs = gy.random_state(int(time.time()*10e7))
            p = self.__gen_prime__(rs)
            q = self.__gen_prime__(rs)
            n = p * q
            lmd = (p - 1) * (q - 1)
            # originally, lmd(lambda) is the least common multiple.
            # However, if using p,q of equivalent length, then lmd = (p-1)*(q-1)
            if gy.gcd(n, lmd) == 1:
                # This property is assured if both primes are of equal length
                break
        g = n + 1
        mu = gy.invert(lmd, n)
        # Originally,
        # g would be a random number smaller than n^2,
        # and mu = (L(g^lambda mod n^2))^(-1) mod n
        # Since q, p are of equivalent length, step can be simplified.
        self.pubKey = [n, g]
        self.priKey = [lmd, mu]
        return

    def decipher(self, ciphertext):
        n, g = self.pubKey
        lmd, mu = self.priKey
        m = self.__L__(gy.powmod(ciphertext, lmd, n ** 2), n) * mu % n
        print("解密的数字:", m)
        plaintext = libnum.n2s(int(m))  # 数字型（不论是十六进制还是十进制）转换为字符串：
        return plaintext

    def encipher(self, plaintext):
        '''加密时取一个随机数 $r$，计算出 $c \equiv g^m r^n(mod\ n^2)$。'''
        m = libnum.s2n(plaintext)  # 文本转换为数字；
        n, g = self.pubKey
        r = gy.mpz_random(gy.random_state(int(time.time())), n ** 2)
        while gy.gcd(n, r) != 1:
            r += 1
        ciphertext = gy.powmod(g, m, n ** 2) * gy.powmod(r, n, n ** 2) % (n ** 2)
        return ciphertext


def test_2():
    pai = Paillier()
    pai.__key_gen__()
    pubKey = pai.pubKey
    priKey = pai.priKey
    print('公钥： {}； 私钥： {}'.format(pubKey, priKey))
    print("公钥，私钥的生成")
    plaintext = input("请输入文本: ")
    # plaintext = 'Cat is the cutest.'
    print("原始文本:", plaintext)
    ciphertext = pai.encipher(plaintext)
    print("加密文本:", ciphertext)
    deciphertext = pai.decipher(ciphertext)
    print("解密文本: ", deciphertext.decode('utf-8'))

def test_ab_2():
    from Pyfhel import Pyfhel, PyPtxt, PyCtxt

    HE = Pyfhel()  # Creating empty Pyfhel object
    HE.contextGen(p=65537)  # Generating context. The p defines the plaintext modulo.
    HE.keyGen()  # Key Generation: generates a pair of public/secret keys
    print(HE)
    integer1 = 127
    integer2 = -2
    ctxt1 = HE.encryptInt(integer1)  # Encryption makes use of the public key
    ctxt2 = HE.encryptInt(integer2)  # For integers, encryptInt function is used.
    print("数据加密")
    print("    int ", integer1, '-> ctxt1 ', type(ctxt1))
    print("    int ", integer2, '-> ctxt2 ', type(ctxt2))

    print(ctxt1)
    print(ctxt2)

    ctxtSum = ctxt1 + ctxt2  # `ctxt1 += ctxt2` for quicker inplace operation
    ctxtSub = ctxt1 - ctxt2  # `ctxt1 -= ctxt2` for quicker inplace operation
    ctxtMul = ctxt1 * ctxt2  # `ctxt1 *= ctxt2` for quicker inplace operation
    print("加密的数据进行 +、-、* ")
    print(f"Sum: {ctxtSum}")
    print(f"Sub: {ctxtSub}")
    print(f"Mult:{ctxtMul}")

    resSum = HE.decryptInt(ctxtSum)  # Decryption must use the corresponding function
    #  decryptInt.
    resSub = HE.decryptInt(ctxtSub)
    resMul = HE.decryptInt(ctxtMul)
    print("加密的数据进行 +、-、* 与明文进行该运算的结果一致：")
    print("     addition:       decrypt(ctxt1 + ctxt2) =  ", resSum)
    print("     substraction:   decrypt(ctxt1 - ctxt2) =  ", resSub)
    print("     multiplication: decrypt(ctxt1 * ctxt2) =  ", resMul)


def main():
    # test_1()
    test_2()


if __name__ == '__main__':
    main()




