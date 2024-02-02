#!/usr/bin/env python
# coding=utf-8

################################################################################################################################
# 方案一(pyhive)：
# 目前实验场景下常见的方案使用pyhive，pyhive通过与HiveServer2通讯来操作Hive数据。当hiveserver2服务启动后，会开启10000的端口，对外提供服务，
# 此时pyhive客户端通过JDBC连接hiveserver2进行Hive sql操作。

# pip3 install pyhive thrift sasl thrift_sasl
from pyhive import hive

conn = hive.Connection(host='30.171.81.72', port=10000, username='abcde', password='ba7bde2', auth='LDAP',database='default')
# host主机ip,port：端口号，username:用户名，database:使用的数据库名称
cursor = conn.cursor()
cursor.execute('SHOW DATABASES')
# 打印结果
for result in cursor.fetchall():
    print(result)

# 关闭连接
cursor.close()
conn.close()

# 注意，若用户名密码未设置，或者错误的话，show databases, 可能查询出结果，但使用 show tables,查询结果为空，且不报错；

################################################################################################################################
# 方案二(impyla)：
# 目前还有用户通过impyla访问hive表，impyla通过与HiveServer2通讯来操作Hive数据。当hiveserver2服务启动后，会开启10000的端口，对外提供服务，
# 此时impyla客户端通过JDBC连接hiveserver2进行Hive sql操作。impyla与hive通信方式和大体相同，具体流程可以参考方案一流程图。

from impala.dbapi import connect

def read_jdbc(host, port, database: str, table: str, query_sql: str) -> DataFrame:
    # 1、连接hive服务端
    conn = connect(host=host, port=10000, database="test", auth_mechanism='PLAIN')
    cursor = conn.cursor()

    # 2、执行hive sql
    cursor.execute(query_sql)
    logger.info('query hive table successfully.')

    # 3、返回pandas.dataframe
    table_len = len(table) + 1
    columns = [col[0] for col in cursor.description]
    col = list(map(lambda x: x[table_len:], columns))
    result = cursor.fetchall()

    return pd.DataFrame(result, columns=col)

################################################################################################################################
# 方案三（pyarrow+thrift）：
# 从方案一流程图中可以了解到上述两种方案都JDBC和服务端建立连接，客户端和hiveserver2建立通信后，解析Hive sql并执行MapReduce的方式访问Hive数据文件，当Hive数据量增大时，对数据进行MapReduce操作和数据之间的网络传输会使得读取数据面临延迟高，效率低等问题。
# 分析上述方案我们可知，在Hadoop集群进行Mapreduce，查询后结果数据经Driver、Executor和hiveserver2才可返回至Client，在数据量级增大的情况下，这些步骤无疑会成为制约python访问hive的效率的因素，为了解决上述问题，我们采用直接读取Hdfs存储文件的方式获取Hive数据的方式，规避上述问题。
#
# hive metastore中存储Hive创建的database、table、表的字段、存储位置等元信息，在读取HDFS文件之前，首先需通过thrift协议和hive metastore服务端建立连接，获取元数据信息；
# 为了解决数据快速增长和复杂化的情况下，大数据分析性能低下的问题，Apache Arrow应运而生，在读取HDFS文件时采用pyarrow读取hive数据文件的方式。

# 为了在本地生成hive metastore服务端文件，首先在hive源码中下载hive_metastore.thrift文件，在thrift源码中下载fb303.thrift文件，其次执行以下命令。
# thrift -gen py fb303.thrift
# thrift -gen py hive_metastore.thrift

# 执行后可以得到以下目录文件
# python向hive表中写入数据和读取hive表的Demo:
from hive_service import ThriftHive
from hive_service.ttypes import HiveServerException
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
import subprocess

from pyarrow.parquet import ParquetDataSet
import pyarrow.parquet as pq
import pyarrow as pa
from libraries.hive_metastore.ThriftHiveMetastore import Client


def connect_hive() -> Client:
    """
    通过thrift连接hive metastore服务端
    """
    transport = TSocket.TSocket(host, int(port))
    transport = TTransport.TBufferedTransport(transport)
    transport.open()
    protocol = TBinaryProtocol.TBinaryProtocol(transport)
    return ThriftHiveMetastore.Client(protocol)


def write_table(client: Client, database: str, table: str, dataframe: DataFrame, partitions: list = None):
    """
    提供给用户将dataFrame写入hive表中的方式

    Examples:
        client = connect_hive(host, port)
         df = pd.DataFrame({
            'index': [1, 2, 3],
            'name': ['xiaoming', 'xiaowang', 'xiaozhang'],
            'prt_dt': ['2020', '2019', '2020']
        })

        partition_cols = ['prt_dt']
        write_table(client, database, table, df, partition_cols)

    Args:
    client(Client):hive客户端，通过thrift协议访问hive metastore
    database(str):数据库
    table(str):表名
    dataframe(pandas.DataFrame):pandas.DataFrame
    partitions(list):分区信息

    raise:
        HiveDatabaseNOTEXIST:Hive库不存在时抛出异常
        HiveTableNOTEXIST:Hive表不存在时抛出异常

    """
    # 1、连接hive服务端
    client = connect_hive(host, port)

    # 2、检查数据库是否存在，如果不存在则抛出异常
    databases = client.get_all_databases()
    if database not in databases:
        raise HiveDatabaseNOTEXIST('Hive database is not exist.')

    # 3、创建hive表，如果表名重复则抛出异常
    tables = client.get_all_tables(database)
    if table not in tables:
        raise HiveTableNOTEXIST('Hive table is not exist.')

    # 4、将pandas中字段int64类型转为int
    columns = dataframe.columns
    int64_fields = {}
    float64_fields = {}
    for field in columns:
        if pd.api.types.is_int64_dtype(dataframe[field]):
            int64_fields[field] = 'int32'

        if pd.api.types.is_float_dtype(dataframe[field]):
            float64_fields[field] = 'float32'
    transfer_fields = dict(int64_fields, **float64_fields)
    transfer_df = dataframe.astype(transfer_fields)

    # 5、将dataframe写入hive表中
    table_hdfs_path = client.get_table(database, table).sd.location
    table = pa.Table.from_pandas(transfer_df)
    pq.write_to_dataset(
        table=table, root_path=table_hdfs_path, partition_cols=partitions)

    # 6、写入分区表时需刷新元数据信息(msck repair table ***)
    shell = "hive -e 'msck repair table {}' ".format('train_data.telecom_train')
    subprocess.Popen(shell, shell=True)


def read_table(data_source: DataSource, database: str, table: str, partitions: list = None) -> DataFrame:
    """
    提供给用户根据hive库名和表名访问数据的方式-->dataframe（thrift、urllib、pyarrow、pyhdfs）

    Examples:
        client = connect_hive(host, port)
        read_table(client,'test','test')

    Args:
        client(Client):hive客户端，通过thrift协议访问hive metastore
        database(str):hive库名
        table(str):hive表名
        partitions(list):hive表分区（用户需按照分区目录填写）,如果查询所有数据，则无需填写分区

    Return:
        pandas.dataframe
    """
    # 1、连接hive服务端
    client = connect_hive(host, port)

    # 2、查询hive表元数据
    table = client.get_table(database, table)
    table_hdfs_path = table.sd.location

    logging.info('table_hdfs_path:' + table_hdfs_path)
    print(table_hdfs_path)

    # 3、判断hive是否为分区表,当用户没有输入partitions时需查找所有分区数据
    if partitions is not None:
        table_hdfs_path = [
            table_hdfs_path + constant.FILE_SEPARATION + x for x in partitions][0]
        dataframe = pq.ParquetDataset(
            table_hdfs_path).read().to_pandas()
        # pyarrow访问分区目录时，dataframe不含分区列，因此需添加分区列信息
        for partition in partitions:
            index = partition.find('=')
            field = partition[:index]
            field_value = partition[index + 1:]
            dataframe[field] = field_value
    else:
        dataframe = pq.ParquetDataset(
            table_hdfs_path).read().to_pandas()
    return dataframe

################################################################################################################################
# 方案对比
# 为了验证分析三种方案在读取数据性能的差异，我们设置了对比实验，准备27维数据，在数据量不断递增情况下执行SELECT查询语句。
# 方案1、方案2效果差不多；
# 数据量小于3万时方案1、方案2在读取效率上优于pyarrow+thrift方案，此后，随着数据量级不断增大，pyarrow+thrift方案较其他两种方案有明显优势。
# 在线下测试中我们发现，读取百万级数据时，pyhive和impyla需要大约4分钟，而pyarrow+thrift只需20s。

# 结论
# 三种方案在读取同一数据时性能上的差异，可以清楚知道数据量在3w左右时，三种方案在读取数据性能上的表现相差不大，但当数据量级不断增大时，通过pyarrow+thrift方案在读取性能上明显优于前两种方案。
# 因此，在万级数据以上推荐使用pyarrow+thrift方式访问Hive数据，可以极大提高python读取hive数据的效率。

# 资料来源：https://blog.csdn.net/qq_29425617/article/details/114451558

# pyhive连接时候报错：
# Error in sasl_client_start (-4) SASL(-4): no mechanism available: No worthy mechs found
# 解决方法：
# 如果是centos系统中需要增加：
# yum install cyrus-sasl-plain  cyrus-sasl-devel  cyrus-sasl-gssapi
#
# 如果是Ubuntu18.04系统中，则需要：
# sudo apt install libsasl2-modules-gssapi-heimdal
#
# Ubuntu中sasl详细参考官网：
# https://packages.ubuntu.com/source/bionic/cyrus-sasl2

def main():
    pass


if __name__ == "__main__":
    main()
