#!/bin/bash

files={
'upload':('libmsc.csv',open("/home/gswewf/graph_qa/input/保单查询标注意图.csv",'rb')) 
}
r=requests.post('http://192.168.3.145:18700/file?pid=zdal'
,files=files) 
r.json()
Out[62]: {'code': 0, 'msg': '文件上传成功！', 'pid': 'zdal'}
files={
'upload':('保单查询标注意图.csv',open("/home/gswewf/graph_qa/input/保单查询标注意图.csv",'rb')) 
}
r=requests.post('http://192.168.3.145:18700/file?pid=zdal'
,files=files) 
r.json()
Out[64]: {'code': 1, 'data': '', 'msg': '服务器错误'}


# filename 项不能为中文，否则tornado服务器解析会出错

def request(method, url, **kwargs):
    """Constructs and sends a :class:`Request <Request>`.

    ...
    :param files: (optional) Dictionary of ``'name': file-like-objects``
        (or ``{'name': file-tuple}``) for multipart encoding upload.
        ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``,
        3-tuple ``('filename', fileobj, 'content_type')``
        or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``,
        where ``'content-type'`` is a string
        defining the content type of the given file
        and ``custom_headers`` a dict-like object 
        containing additional headers to add for the file.
The relevant part is: file-tuple can be a2-tuple, 3-tupleor a4-tuple.


