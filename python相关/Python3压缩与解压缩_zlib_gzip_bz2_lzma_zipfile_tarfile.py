#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 原文: http://blog.csdn.net/Rozol/article/details/72672703

# 文件的归档(各种格式的压缩 / 解压缩)

import zlib


def zlib_demo():
    # 对直接进行压缩
    data = b'luzhuo.me'

    com_bytes = zlib.compress(data)
    print("压缩后的数据: ", com_bytes)

    decom_bytes = zlib.decompress(com_bytes)
    print("解压后的数据: ", decom_bytes)

    # 对数据流进行压缩
    with open("file.txt", "rb") as read, open("com.txt", "wb") as write:
        com = zlib.compressobj(level=9, memLevel=9)
        for data in read:
            # 压缩数据并写入文件
            write.write(com.compress(data))
        write.write(com.flush())

    # 对数据流进行解压
    with open("com.txt", "rb") as read, open("temp.txt", "wb") as write:
        decom = zlib.decompressobj()
        for data in read:
            write.write(decom.decompress(data))
        write.write(decom.flush())


def zlib_func():
    '''
    zlib
    '''

    # zlib.adler32(data[, value]) // 计算数据的Adler-32校验和(比CRC32快), value:校验和的起始值(默认1)
    num = zlib.adler32(b"luzhuo.me")
    # zlib.crc32(data[, value]) // 计算数据的CRC(循环冗余校验)校验和, value:校验和的起始值,默认0
    num = zlib.crc32(b"luzhuo.me")

    # zlib.compress(data[, level])  // 压缩字节数据,返回压缩后的字节, level:[0,9],0:无压缩,1最小压缩,9最高压缩, 默认:6
    bytes = zlib.compress(b"luzhuo.me")
    # zlib.decompress(data[, wbits[, bufsize]])  // 解压缩, wbits:历史缓冲区(默认:15), bufsize:保存解压数据的缓冲区(默认:16384)
    bytes = zlib.decompress(bytes)

    # 压缩对象 level:压缩级别[0,9], method:压缩算法(DEFLATED), wbits:历史缓冲区大小,值越大,压缩效果更好(注:使用默认值), memLevel:内存量[1,9],值越大,效果更好更快
    # strategy:策略 Z_DEFAULT_STRATEGY / Z_FILTERED / Z_HUFFMAN_ONLY, zdict:压缩字典(预期频繁出现的字节序列)
    # zlib.compressobj(level=-1, method=DEFLATED, wbits=15, memLevel=8, strategy=Z_DEFAULT_STRATEGY[, zdict])
    com = zlib.compressobj(level=9, memLevel=9)
    # zlib.decompressobj(wbits=15[, zdict]) // 解压缩对象
    decom = zlib.decompressobj()

    strs = zlib.ZLIB_VERSION  # zlib版本
    strs = zlib.ZLIB_RUNTIME_VERSION  # 解释器运行的zlib版本

    # 压缩对象
    com_data = com.compress(b"luzhuo.me")  # 压缩数据
    # Compress.flush([mode]) // 完成剩余压缩数据的字节对象, mode:Z_SYNC_FLUSH(可进步压缩) / Z_FULL_FLUSH(可进步压缩) / Z_FINISH(完成压缩)
    data = com.flush()
    com.copy()  # 拷贝压缩对象副本

    # 解压缩对象
    # Decompress.decompress(data[, max_length])  // 解压
    data = decom.decompress(data)

    decom.unused_data  # 将要解压的数据
    decom.unconsumed_tail  # 未解压的数据
    decom.eof  # 数据流是否已结束
    decom.flush()  # 完成压缩, 无法再次解压, length:缓冲区大小
    decom.copy()

    # 异常
    try:
        pass
    except zlib.error:
        pass  # 压缩和解压缩错误而引发的异常



# ==================================


import gzip
import shutil

content = b"luzhuo.me"


def gzip_demo():
    # 类字节对象的压缩与解压
    # 压缩
    bytes_com = gzip.compress(content)
    print("字节压缩: ", bytes_com)
    bytes_decom = gzip.decompress(bytes_com)
    print("字节解压: ", bytes_decom)

    # 对gzip文件的读写操作
    # 写入
    with gzip.open('box.gz', 'wb') as write:
        write.write(content)
    # 读取
    with gzip.open('box.gz', 'rb') as read:
        data = read.read()
        print(data)

    # 文件对象的压缩与解压
    # 压缩
    with open('file.txt', 'rb') as read, gzip.open('file.txt.gz', 'wb') as write:
        shutil.copyfileobj(read, write)
    # 解压
    with gzip.open('file.txt.gz', 'rb') as read, open('temp.txt', 'wb') as write:
        shutil.copyfileobj(read, write)


def gzip_func():
    '''
    该模块提供简单压缩和解压缩文件, 数据压缩有zlib提供
    f = gzip.open(), 压缩就直接往里写, 解压就直接读取
    '''

    # 打开gzip压缩文件: filename:文件名或file对象, mode:读写模式,二进制: 'r', 'rb'(默认), 'a', 'ab', 'w', 'wb', 'x', 'xb',文本:'rt', 'at', 'wt', 'xt', compresslevel:压缩级别[0,9], 文本模式可提供(二进制模式不需要):encoding / errors / newline
    # gzip.open(filename, mode='rb', compresslevel=9, encoding=None, errors=None, newline=None)
    file = gzip.open("box.gz")

    # gzip.compress(data, compresslevel=9) // 压缩
    bytes = gzip.compress(b"luzhuo.me")
    bytes = gzip.decompress(bytes)  # 解压



# ================================


import bz2


def bz2_demo(content):
    # 单次压缩
    bytes_com = bz2.compress(content.encode('utf-8'))
    print("单次压缩: ", bytes_com)
    bytes_decom = bz2.decompress(bytes_com).decode('utf-8')
    print("单次解压: ", bytes_decom)

    # 增量压缩
    bzcom = bz2.BZ2Compressor()
    bzdecom = bz2.BZ2Decompressor()

    bytes_com = bzcom.compress(content)
    bytes_com += bzcom.flush()
    print("增量压缩: ", bytes_com)

    bytes_decom = bzdecom.decompress(bytes_com)
    print("增量解压: ", bytes_decom)

    # 读写压缩
    with open('file.txt', 'rb') as read, bz2.open('file.txt.gz', 'wb') as write:
        shutil.copyfileobj(read, write)
    with bz2.open('file.txt.gz', 'rb') as read, open('temp.txt', 'wb') as write:
        shutil.copyfileobj(read, write)


def bz2_func():
    '''
    使用bzip2压缩算法压缩和解压
    该模块的所有类都是线程安全的
    读取和写入压缩文件: open() BZ2File
    增量压缩: BZ2Compressor BZ2Decopressor
    单次压缩: compress() decompress()
    '''

    # 读写压缩文件
    # 二进制或文本模式打开bzip2压缩文件
    # bz2.open(filename, mode='r', compresslevel=9, encoding=None, errors=None, newline=None)
    file = bz2.open("box.bz2")

    # 单次压缩
    # bz2.compress(data, compresslevel=9)
    bytes_com = bz2.compress(b'luzhuo.me')
    # bz2.decompress(data)
    bytes_decom = bz2.decompress(bytes_com)

    # 增量压缩
    # class bz2.BZ2Compressor(compresslevel=9)
    bzcom = bz2.BZ2Compressor()

    bytes_com = bzcom.compress(b'luzhuo.me')  # 压缩
    bytes_com = bzcom.flush()  # 完成压缩, 返回剩余的压缩数据

    # 增量压缩的解压缩
    # class bz2.BZ2Decompressor
    bzdecom = bz2.BZ2Decompressor()

    # decompress(data, max_length=-1) // 解压
    bytes_decom = bzdecom.decompress(bytes_com)
    boolean = bzdecom.eof  # 是否到达流结束标记


# =====================================

import lzma


def lzma_demo():
    # 增量压缩
    lzmacom = lzma.LZMACompressor()
    data = lzmacom.compress(b'luzhuo')
    data += lzmacom.compress(b'.')
    data += lzmacom.compress(b'me')
    print("增量压缩: ", data)

    # 读写
    with open('file.txt', 'rb') as read, lzma.open('file.txt.xz', 'wb') as write:
        shutil.copyfileobj(read, write)


def lzma_func():
    '''
    LZMA压缩算法压缩和解压缩
    LZMAFile线程不是安全的
    '''

    # 文件
    # 读取:format, filters, 写入:format, check, preset, filters
    # lzma.open(filename, mode="rb", *, format=None, check=-1, preset=None, filters=None, encoding=None, errors=None, newline=None)
    f = lzma.open("box.xz")

    # lzma.compress(data, format=FORMAT_XZ, check=-1, preset=None, filters=None) // 压缩
    bytes_com = lzma.compress(b'luzhuo.me')
    # lzma.decompress(data, format=FORMAT_AUTO, memlimit=None, filters=None) // 解压
    bytes_decom = lzma.decompress(bytes_com)

    # lzma.is_check_supported(check) // 是否支持完整性检查
    boolean = lzma.is_check_supported(lzma.CHECK_SHA256)

    # 压缩对象 check:完整性检查:CHECK_NONE,CHECK_CRC32(32位循环冗余校验),CHECK_CRC64(64位循环冗余校验),CHECK_SHA256(sha256)
    # preset: 压缩级别[0,9]
    # class lzma.LZMACompressor(format=FORMAT_XZ, check=-1, preset=None, filters=None)
    lzmacom = lzma.LZMACompressor()

    bytes_com = lzmacom.compress(b"luzhuo.me")  # 压缩
    bytes_com = lzmacom.flush()  # 完成压缩

    # 解压缩对象
    # class lzma.LZMADecompressor(format=FORMAT_AUTO, memlimit=None, filters=None)
    lzmadecom = lzma.LZMADecompressor()
    # decompress(data, max_length=-1)
    bytes_decom = lzmadecom.decompress(bytes_com)  # 解压

    # 异常
    try:
        pass
    except lzma.LZMAError:
        pass  # 在压缩或解压或初始化时发生错误



# ================================================


import zipfile


def zip_demo():
    # 创建压缩包
    with zipfile.ZipFile(r"file.zip", "a") as write:
        # 往压缩包里添加文件
        write.write("file.txt")

        # 测试压缩包
        error = write.testzip()
        if error:
            print("压缩文件错误: ", error)
            return

        # 打印压缩包信息
        write.printdir()
        write.close()

    # 解压压缩包
    with zipfile.ZipFile(r"file.zip", "r") as read:
        read.extractall("temp")
        read.close()


def zip_func():
    '''
    ZIP格式的压缩和解压
    不支持多磁盘ZIP文件处理
    可以使用ZIP64扩展(>4G的ZIP文件)
    可以解密经过加密的ZIP文件,但是不能创建加密的ZIP文件
    解密速度缓慢,因为是通过Python实现的
    '''

    # class zipfile.ZipFile // 读写的zip类
    # class zipfile.ZipInfo(filename='NoName', date_time=(1980, 1, 1, 0, 0, 0)) // Zip信息

    zipfile.ZIP_STORED  # 存储 方法常量0
    zipfile.ZIP_DEFLATED  # ZIP压缩 方法常量8
    zipfile.ZIP_BZIP2  # BZIP2压缩 方法常量12
    zipfile.ZIP_LZMA  # LZMA压缩 方法常量14

    boolean = zipfile.is_zipfile(r"file.zip")  # 是否是zip文件, 参数可为文件名 / 类文件对象  (测试:.rar也是zip文件)

    # --- class zipfile.ZipFile // 读写的zip类 ---
    # file:文件名 / 类文件对象, mode:r, w, a, x(创建并写入), compression:ZIP_STORED, ZIP_DEFLATED, ZIP_BZIP2, ZIP_LZMA, allowZip64:zip文件大于2G时需要使用, 支持with
    # class zipfile.ZipFile(file, mode='r', compression=ZIP_STORED, allowZip64=True)
    zipf = zipfile.ZipFile(r"file.zip")

    zipf.close()  # 关闭压缩文件,并写入基本信息
    lists = zipf.infolist()  # 压缩包里每个成员文件的zipinfo对象
    lists = zipf.namelist()  # 压缩包里的成员文件列表
    info = zipf.getinfo(lists[0])  # 获取指定成员文件的zipinfo对象
    # ZipFile.open(name, mode='r', pwd=None)  //打开成员文件, name:文件名 / zipinfo, pwd:密码,用于加密了的文件
    f = zipf.open(lists[0])
    # ZipFile.extract(member, path=None, pwd=None) // 提取成员文件, member:文件名 / zipinfo, path:提取到指定目录, 返回提取后的文件绝对路径
    path = zipf.extract(lists[0])
    # ZipFile.extractall(path=None, members=None, pwd=None) // 提取全部成员文件, members:提取的成员,由namelist()返回的列表子集
    zipf.extractall()
    zipf.printdir()  # 打印(sys.stdout)成员文件
    zipf.setpassword("123")  # 设置密码, 用于提取加密文件
    # ZipFile.read(name, pwd=None)  // 读取指定成员文件字节内容, 压缩文件必须以 r / a 打开
    zipf.read(lists[0])
    # ZipFile.write(filename, arcname=None, compress_type=None) // 追加成员文件, filename:文件名, arcname:写入压缩包的名字(默认同filename)
    zipf.write("temp.txt")
    # ZipFile.writestr(zinfo_or_arcname, data[, compress_type]) // 成员文件写入字符串, zinfo_or_arcname: 成员文件名 / zipinfo
    zipf.writestr(lists[0], b"luzhuo.me")  # 压缩文件必须以 w, a, x 打开, 覆盖写入成员文件, 并且最后要close()
    zipf.testzip()  # 测试压缩包, 检查CRC和文件头, 返回第一个坏文件的名称, 没有坏文件返回None
    zipf.debug  # 测试输出, [0,3], 0(默认):无输出, 3:最大输出
    zipf.comment  # 注释文本, 压缩文件以 w, x ,a 打开, 不超过65535字节的字符

    # ZipInfo类的实例由ZipFile对象的getinfo()和infolist()方法返回
    # --- class zipfile.ZipInfo(filename='NoName', date_time=(1980, 1, 1, 0, 0, 0)) // Zip信息 ---
    info.filename  # 成员文件文件名
    info.date_time  # 修改日期时间 (年, 月, 日, 时, 分, 秒)
    info.compress_type  # 压缩类型
    info.comment  # 注释
    info.compress_size  # 文件压缩后的大小
    info.file_size  # 文件未压缩时的大小

    info.extra  # 扩展字段数据
    info.create_system  # 创建ZIP存档的系统
    info.create_version  # PKZIP版本创建ZIP存档
    info.extract_version  # PKZIP版本需要提取归档
    info.reserved  # 0
    info.flag_bits  # ZIP标志
    info.volume  # 文件头的卷编号
    info.internal_attr  # 内部属性
    info.external_attr  # 外部文件属性
    info.header_offset  # 字节偏移到文件头
    info.CRC  # CRC - 32的未压缩文件

    # 异常
    try:
        pass
    except zipfile.BadZipFile:
        pass  # zip文件引发的错误
    except zipfile.LargeZipFile:
        pass  # 需要ZIP64功能, 但未启用

# ==============================================


import tarfile


def tarfile_demo():
    # 创建压缩包
    with tarfile.open(r"file.tar", "w:gz") as write:
        # 往压缩包里添加文件
        write.add("file.txt")

        # 测试压缩包
        # 没有提供测试函数

        # 打印压缩包信息
        write.list()

    # 解压压缩包
    with tarfile.open(r"file.tar", "r:gz") as read:
        read.extractall("temp")


def tarfile_func():
    '''
    tar文件的压缩和解压
    同时支持gzip / bz2 / lzma的相关操作
    '''

    # mode:
    # 'r''r：*': 打开(默认)
    # 'r:': 无压缩打开
    # 'r:gz': gzip压缩打开
    # 'r:bz2': bzip2压缩打开
    # 'r:xz': lzma压缩打开
    # 'x''x:': 创建无压缩tarfile
    # 'x:gz': 创建gzip压缩tarfile
    # 'x:bz2': 创建bzip2压缩tarfile
    # 'x:xz': 创建lzma压缩tarfile
    # 'a''a：': 打开
    # 'w''w：': 未压缩的写入(注:存在则覆盖)
    # 'w:gz': gzip压缩写入
    # 'w:bz2':bzip2压缩写入
    # 'w:xz': lzma压缩写

    # 'r|*': 读取流
    # 'r|': 读取未压缩流
    # 'r|gz': 读取gzip压缩流
    # 'r|bz2': 读取bzip2压缩流
    # 'r|xz': 读取lzma压缩流
    # 'w|': 写入未压缩流
    # 'w|gz': 写入gzip压缩流
    # 'w|bz2': 写入bzip2压缩流
    # 'w|xz': 写入lzma压缩流

    tarfile.ENCODING  # 编码
    tarfile.USTAR_FORMAT  # POSIX.1 - 1988格式
    tarfile.GNU_FORMAT  # GNU tar格式
    tarfile.PAX_FORMAT  # POSIX.1 - 2001格式
    tarfile.DEFAULT_FORMAT  # GNU_FORMAT(默认格式)

    # tarfile.open(name=None, mode='r', fileobj=None, bufsize=10240, **kwargs) // fileobj: 二进制file-object  支持with
    tarf = tarfile.open(r"file.tar")

    tarfile.is_tarfile(r"file.tar")  # 是否是tar文件

    # class tarfile.TarFile // tarfile.open()返回的对象
    # mode: r, a, w, x, fileobj:file-obj, format:USTAR_FORMAT,GNU_FORMAT,PAX_FORMAT, tarinfo:可替换默认的TarInfo, dereference:True添加文件,False添加软硬链接, ignore_zeros: 是否忽略空块(损坏的文件设为False), debug:[0,3]
    # class tarfile.TarFile(name=None, mode='r', fileobj=None, format=DEFAULT_FORMAT, tarinfo=TarInfo, dereference=False, ignore_zeros=False, encoding=ENCODING, errors='surrogateescape', pax_headers=None, debug=0, errorlevel=0)

    # 类方法
    tarfile.TarFile.open(...)  # 同 tarfile.open()

    tarinfo = tarf.getmember("file.txt")  # 获取指定成员文件的TarInfo对象, 未找到KeyError
    lists = tarf.getmembers()  # 成员文件TarInfo列表
    lists = tarf.getnames()  # 成员文件名字列表
    # TarFile.list(verbose=True, *, members=None) // 打印目录, verbose:是否详细, members:可选成员,getmembers()的子集
    tarf.list()
    tarf.next()  # 下个文件的TarInfo
    # TarFile.extractall(path=".", members=None, *, numeric_owner=False) // 解压全部
    tarf.extractall()
    # TarFile.extract(member, path="", set_attrs=True, *, numeric_owner=False) // 解压指定成员文件
    tarf.extract(lists[0])
    bf_read = tarf.extractfile(lists[0])  # 提取成员文件,返回io.BufferedReader对象
    # name: 文件名, arcname:存储的成员文件名, exclude: def exclude(filename){已弃用}:return True排除,False添加, filter:def filter(tarinfo):return Tarinfo添加,None排除
    # TarFile.add(name, arcname=None, recursive=True, exclude=None, *, filter=None)
    tarf.add("temp.txt")
    # TarFile.addfile(tarinfo, fileobj=None) // 添加tarinfo
    tarf.addfile(tarf.gettarinfo())
    # TarFile.gettarinfo(name=None, arcname=None, fileobj=None) // 获取TarInfo,可通过addfile()修改, 非r模式
    tarinfo = tarf.gettarinfo(arcname="file.txt")
    tarf.close()  # 关闭, 并写入两个零块

    tarf.pax_headers  # pax全局头的键值对的字典

    # --- TarInfo ---
    # 创建
    # class tarfile.TarInfo(name="")
    tarinfo = tarfile.TarInfo(name="temp.txt")

    # 类方法
    # TarInfo.frombuf(buf, encoding, errors) // 从字符缓冲区创建TarInfo
    tarinfo = tarfile.TarInfo.fromtarfile(tarf)  # tarfile读取下个成员,返回TarInfo
    # TarInfo.tobuf(format=DEFAULT_FORMAT, encoding=ENCODING, errors='surrogateescape') // 从TarInfo创建字符缓冲区

    tarinfo.name  # 文件名
    tarinfo.size  # 大小(bite)
    tarinfo.mtime  # 修改时间戳
    tarinfo.mode  # 权限

    tarinfo.type  # 类型:REGTYPE,AREGTYPE,LNKTYPE,SYMTYPE,DIRTYPE,FIFOTYPE,CONTTYPE,CHRTYPE,BLKTYPE,GNUTYPE_SPARSE
    tarinfo.linkname  # 目标文件名
    tarinfo.uid  # 用户id
    tarinfo.gid  # 用户组
    tarinfo.uname  # 用户名
    tarinfo.gname  # 用户组名
    tarinfo.pax_headers  # pax扩展头的键值对的字典

    tarinfo.isfile()  # 是否是文件
    tarinfo.isreg()  # 同isfile
    tarinfo.isdir()  # 是否是目录
    tarinfo.issym()  # 是否是符号链接
    tarinfo.islnk()  # 是否是硬链接
    tarinfo.ischr()  # 是否是字符设备
    tarinfo.isblk()  # 是否是块设备
    tarinfo.isfifo()  # 是否是FIFO
    tarinfo.isdev()  # 是否是字符设备，块设备或FIFO之一

    # 异常
    try:
        pass
    except tarfile.TarError:
        pass  # tarfile异常的基类
    except tarfile.ReadError:
        pass  # tar文件打开时引发的异常
    except tarfile.CompressionError:
        pass  # 无法解码
    except tarfile.StreamError:
        pass  # 数据流异常
    except tarfile.ExtractError:
        pass  # TarFile.extract()(解压)引发的异常
    except tarfile.HeaderError:
        pass  # TarInfo.frombuf()(缓冲区无效)引发的异常

    # 命令行模式(python -m tarfile -l file.tar)
    # -l <tarfile>
    # --list <tarfile>
    # 列出成员文件
    #
    # -c <tarfile> <source1> <sourceN>
    # --create <tarfile> <source1> <sourceN>
    # 创建, tarfile:压缩包名, source1:资源名...
    #
    # -e <tarfile> [<output_dir>]
    # --extract <tarfile> [<output_dir>]
    # 解压, output_dir:解压目录
    #
    # -t <tarfile>
    # --test <tarfile>
    # 测试
    #
    # -v
    # --verbose
    # 打印命令详情


def test():
    zlib_demo()
    zlib_func()

    gzip_demo()
    gzip_func()

    bz2_demo('你好')
    bz2_func()

    lzma_demo()
    lzma_func()

    zip_demo()
    zip_func()

    tarfile_demo()
    tarfile_func()

    # 使用示例：
    code = "QlpoOTFBWSZTWXLbSoQAAgzbgERwQXxmTwAAr/ff3kABt0Q2oRVT0hpo9RtEAAAAyBEiSQ9EGjQGQAAAwANGhowjJoNGmgMEUplMTNSMJ5TQaDJpsoMyRMj8P4mZzFSVVwqSXG8GG7MlVwiToYEQwVD7noBxMhNfkeZYtYFtbgOBUSIGtIQjhNHCEnPJsadhb3yBmRIOD3TeAtNLSaU5GgvKUBWSNuuOIHmVt0YhW6rsmDMDUjeUJGJ64R1Jm5lrh0Aa0tKjhFwPdWcGogxLDSXPWQUWTM8Sd3Qz1HMYNxx3HMeiNqNo4jeRDEfZ3gUSHIcU/heomq0vEzL1Msz5KKGxH8FrNOYw3KaxdqaEmNHYMxJFgQbR0DyRknL2L4kwUSxKRdhjRpEtUqilVfggFL1klaMS3PPRDfNqbBOPWO7m4JTVGhS9QTBDDJaEbLbrUQNB+IpJSKQbG5SZZ5gkwJEhJ3aYKJipZ/i7kinChIOW2lQg"
    cpu_quantization_code = bz2.decompress(base64.b64decode(code)).decode()
    old_code = base64.b64encode(bz2.compress(cpu_quantization_code.encode('utf-8'))).decode('utf-8')
    print(cpu_quantization_code, old_code==code)

def main():
    test()


if __name__ == '__main__':
    main()
