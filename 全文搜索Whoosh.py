#! /usr/lib/python3
# -*- coding: utf-8 -*-

#http://blog.csdn.net/twsxtd/article/details/8308893

最近想做一个搜索引擎，当然少不了看下闻名遐迩的Lucene，不得不说确实非常出色，但是对于python的实现pylucene确是差强人意，首先它 不是纯python实现
而是做了一层包装到头来还是使用java，依赖于JDK不说安装步骤繁琐至极，而且Lucene可用的中文分词词库非常之多但是由 于这层粘合关系很多都用不上，
最终还是放弃，不过平心而论如果用Java实现的确很完美。其它的有sphinx以及基于它实现的专门用于中文的 coreseek，不过看了下好像是基于SQL语言的，
对于我要做的事情好像关系不大；还有用C++写的xapian框架，可以说是一片好评啊，速度精度 都非常不错，但最终还是看上了纯python实现的Whoosh，
首先对于python使用来说非常简单，就是一个模块，easy_install就行， 但是搜了一下国内的资料非常之少，没有办法，就把它的文档翻译一下吧~~今天开始

Quick Start
    Whoosh是一个索引文本和搜索文本的类库，他可以为你提供搜索文本的服务，比如如果你在创建一个博客的软件，你可以用whoosh为它添加添加一个搜索功能以便用户来搜索博客的入口
下面是一个简短的例子：
from whoosh.index import create_in
from whoosh.fields import *
schema = Schema(title = TEXT(stored = True),path = ID(stored=True),content=TEXT)
ix = create_in("/home/gswyhq/百科/indexer",schema)#（这里的“indexer”实际上是一个目录，因此按照这个步骤来会出错，你得先创建目录，译者注）
writer = ix.writer()
writer.add_document(title=u"First document",path=u"/a",
                    content = u"this is the first document we've add!")
writer.add_document(title=u"Second document", path=u"/b",
                        ...                     content=u"The second one is even more interesting!")
writer.commit()
from whoosh.qparser import QueryParser
with ix.searcher() as searcher:
    query = QueryParser("content", ix.schema).parse("first")
    results = searcher.search(query)
    results[0]
{"title": u"First document", "path": u"/a"}

Index和Schema对象

在 开始使用whoosh之前，你需要一个index对象，在你第一次创建index对象时你必须定义一个Schema对象，Schema对象列出了 Index的所有域。
一个域就是Index对象里面每个document的一个信息，比如他的题目或者他的内容。一个域能够被索引（就是能被搜索到）或者 被存储（就是得到索引之后的结果，
这对于标题之类的索引非常有用）
下面这个schema对象有两个域，“title”和“content”

from whoosh.fields import Schema,TEXT
schema = Schema(title=TEXT,content=TEXT)

当你创建index的时候你创建一次schema对象就够了，schema是序列化并且和index存储的。
当你创建索引对象的时候，需要用到关键字参数来将域名和类型建立映射关系，域的名字和类型决定了你在索引什么和什么事可搜索的。whoosh有很多非常有用的
预定义域类型，你可以非常容易的创建他们。如下：
whoosh.fields.ID
这种类型简单地索引（选择性存储）这个域里面所有的值作为一个单元（意思就是它不会把里面的词分开）这个对于像存储文件路径，URL，日期等等非常有用
whoosh.fileds.STROED
这个域存储文本但是不索引，他不被索引也不可被搜索，当你需要给用户展示文本搜索结果信息的时候非常有用
whoosh.fileds.KEYWORD
这种类型是为空格或者逗号分词的关键字设计的。他可以被索引和被搜索到（并且选择性的存储）。为了保存空格他不支持词组搜索
whoosh.fields.TEXT
这是正文文本，他索引（并且选择性存储）文本并且存储词语位置来支持短语搜索
whoosh。fields.NUMERIC
数字类型，可以存储整数和浮点数
whoosh.fields.BOOLEAN
布尔类型
whoosh.fields.DATETIME
日期类型，以后有详细介绍
whoosh.fields.NGRAM和whoosh.fields.NGRAMWORDS
这种类型把文本分成n-grams（以后详细介绍）
（为了简便，你可以不给域类型传递任何参数，你可以仅仅给出类型名字whoosh会为你创造类的实例）

from whoosh.fields import Schema,STROED,ID,KEYWORD,TEXT

schema = Schema(title=TEXT(stored=True),content=TEXT,
                path=ID(stored=True),tags=KEYWORD,icon=STROED)

一旦你有了schema对象，你可以用create_in函数创建索引：

import os.path
from whoosh.fields import create_in

#os.path.exists(path) 如果path存在,返回True;如果path不存在,返回False
if not os.path.exists("index"):
    os.mkdir("index")
    ix = create_in("index",schema)

(这创造了一个存储对象来包含索引，存储对象代表存储索引的媒介，通常是文件存储，也就是存在目录的一系列文件)
在你创建索引之后，你可以通过open_dir函数方便地打开它

from whoosh.index import open_dir

ix = open_dir("index")

IndexWriter对象
好了我们有了索引对象现在我们可以添加文本。索引对象的writer()方法返回一个能够让你给索引添加文本的IndexWriter对象，IndexWriter对象的
add_document(**kwargs)方法接受关键字参数：
writer = ix.writer()
writer.add_document(title=u"My document", content=u"This is my document!",
                    path=u"/a",tags=u"first short", icon=u"/icons/star.png")
writer.add_document(title=u"Second try", content=u"This is the second example.",
                    path=u"/b", tags=u"second short", icon=u"/icons/sheep.png")
writer.add_document(title=u"Third time's the charm", content=u"Examples are many.",
                    path=u"/c", tags=u"short", icon=u"/icons/book.png")
writer.commit()

两个重点：
1.你不必要把每一个域都填满，whoosh不关系你是否漏掉某个域
2.被索引的文本域必须是unicode值，被存储但是不被索引的域（STROED域类型）可以是任何和序列化的对象
如果你需要一个既要被索引又要被存储的文本域，你可以索引一个unicode值但是存储一个不同的对象（某些时候非常有用）
writer.add_document(title=u"Title to be indexed",_stored_title=u"Stored title")

使用commit()方法让IndexWriter对象将添加的文本保存到索引
writer.commit()

Searcher对象

在搜索文本之前，我们需要一个Searcher对象
searcher = ix.searcher()

推荐用with表达式来打开搜索对象以便searcher对象能够自动关闭（searcher对象代表打开的一系列文件，如果你不显式的关闭他们，系统会慢慢回收他们以至于有时候会用尽文件句柄）

with ix.searcher() as searcher:
    ...

这等价于：

try:
    searcher = ix.searcher()
    ...
finally:
    searcher.close()

Searcher对象的search方法使用一个Query对象，你可以直接构造query对象或者用一个query parser来得到一个查询语句

#Construct query objects directly

from whoosh.query import *
myquery = And([Term("content",u"apple"),Term("content","bear")])

用parser得到一个查询语句，你可以使用默认的qparser模块里面的query parser。QueryParser构造器的第一个参数是默认的搜索域，这通常是“正文文本”域第二个可选参数是用来理解如何parse这个域的schema对象：

#Parse a query string

from whoosh.qparser import QueryParser
parser = QueryParser("content",ix.schema)
myquery = parser.parse(querystring)

一旦你有了Searcher和query对象，你就可以使用Searcher对象的search()方法进行查询并且得到一个结果对象

>>>results = searcher.search(myquery)
>>>print len(results)
1
>>>print results[0]
{"title": "Second try", "path": /b,"icon":"/icon/sheep.png"}

默认的QueryParser实现了一种类似于lucene的查询语言，他允许你使用连接词AND和OR以及排除词NOT和组合词together来搜索，他默认使用AND将从句连结起来（因此默认你所有的查询词必须都在问当中出现）

>>> print(parser.parse(u"render shade animate"))
And([Term("content", "render"), Term("content", "shade"), Term("content", "animate")])

>>> print(parser.parse(u"render OR (title:shade keyword:animate)"))
Or([Term("content", "render"), And([Term("title", "shade"), Term("keyword", "animate")])])

>>> print(parser.parse(u"rend*"))
Prefix("content", "rend")

Whoosh处理结果包含额外的特征，例如：
1.按照结果索引域的值排序，而不是按照相关度
2.高亮显示原文档中的搜索词
3.扩大查询词给予文档中找到的少数值
4.分页显示（例如显示1-20条结果，第一到第四页的结果）

关于如何搜索后面有详细介绍

好，第一篇到此为止，后面会陆陆续续地翻译出来，算是开个头吧


术语

Analysis
把文本分成单个词语并且索引的过程，先把文本分成词组，然后选择性的过滤（大小写之类的）Whoosh包含几种不同的analyzer

Corpus
你正在索引的文本集

Documents
你希望能有搜索到的单个词的内容,"documents"可能是有文件的意思，但是事实上数据源可以时任意的——管理系统上面的文章，博客系统上面的博客，一个大文件的文件块，SQL查询的一行数据，个人邮件或者任何别的东西。当你从Whoosh得到搜索结果的时候，结果是一个documents的列表，无论“document”在你的搜索引擎里面意味着什么

Fields
每一个document包含着一个Field集，典型的Filed可能是“title”，“content”,“url”，“keyword”，“status”，“data”等等。Fields可以被索引（也就是可以被搜索到）并且和document存储，存储field可以在搜索结果中使用，比如你可能希望存储title这个field以便都所结果可以显示它

Forward index
一个陈列着在document中出现的每一个document和words的表，Whoosh存储的term vertors就是一种forward index

Indexing
检测在文本集中的documents并且把他们添加到反向索引的过程

Postings
反向索引列出的出现在corpus中出现的每一个单词，对于每个单词来说，和这个单词一起出现的documents以及伴随这的选择性的信息（例如出现的次数）。这写出现在列表中的东西，包含着document的数量以及人和的额外信息叫做postings。在Whoosh里面存储在postings里面的每个field的信息是可定制的

Resverse index
基本上就是列出了每个corpus里面的word，然后列出了每个与这个word相关的document的表，它可以更复杂（可以列出每个word出现在document中出现的次数以及每次出现的位置）但是这仅仅是他基本的工作方式

Schema
Whoosh在你开始indexing之前需要你列举索引的每一个field，Schema把你的field的名字和他们的元数据联系起来，比如postings的格式以及他们是否被存储

Term vector
在一个特定的document里面某个特定field的forward index，你可以指明在Schema中某个域必须存储term vector。






About schema and fields

schema指明了在一个index中的document的field
每个document可以有多个field，比如title，content，url，data等等
有些field可以被索引，有些field可以和document一起存起来以便使得在搜索结果的时候可以显示出来。有些field既可以被索引也可以被存储。
schema就document中所有可能field的集合，每一个document可能仅仅使用schema中field的某个子集，例如，一个简单的检索邮件的schema可能像这样：
from_addr,to_addr,subject,body,和attachments,attachments列出与这封邮件相关的邮件，对于没有相关邮件的你可以省略它

Built-in field types(内建field类型)

Whoosh提供了很多非常有用的与定义field类型：

whoosh.fields.TEXT

这种类型是正文文本，它索引文本（并且选择性地存储）并且存储位置项以便搜索
TEXT使用 StandardAnalyzer吗？默认是这样。当然也可以指明不同的分词器，给构造器一个关键词参数就可以
TEXT(analyzer=analysis.StemmingAnalyzer()),看看TextAnalysis？
TEXT field存储每一个索引项的位置信息以便可以短语搜索，如果你不需要，你可以关闭它以节省空间：TEXT（parse=False）
TEXT默认不存储，通常你不希望在索引里面存储正文信息，通常你有被索引的document给就诶过提供链接，因此你不需要在索引里面存储它。然而，在某些情况下面他们可能有用，用TEXT（stored=True）来指明这写文本需要被存储

whoosh.fields.KEYWORD

这种类型为空格或者逗号分词设计。它是被索引并且可搜索的（选择性存储），为了节省空间他不支持短语搜索
为了在索引里面存储值，可以在构造器里面使用 stored=True，可以使用 lowercase=True来使文本自动变成小写
默认是空格分词的，可以用commas=True来使用commas分词（允许空格）
如果用户要使用keyword都所，使用scorable=True

whoosh.fields.ID

ID field类型简单地索引这个field里面的整个值作为一个单元（也即它不会分成若干项）这种类型不存储出现频率信息，因此它十分紧凑，但不适合计数
像文件路径，URL，data或者catalog（必须当成一个整体并且每个document只有一个值）的时候可以使用ID这个域
默认ID是不存储的，使用ID（stored=True）指明需要存储，例如你可能需要存储url以便在结果中可以提供链接

whoosh.fields.STORED
这个field和document一并存储，但是不可索引和不可搜索，这在你提供搜索结果的更多信息但是不需要搜索它的时候有用

whoosh.fields.NUMERIC
这个field以一种紧凑可排序的格式存储整型，长整型，实型数据

whoosh.fields.DATETIME
这个field以一种紧凑可排序的格式存储日期型数据

whoosh.fields.BOOLEAN
这个field简单地存储布尔值并且支持用户用 yes,no,true,false,1,0,t或者f搜索

whoosh.fields.NGRAM
TBD.
专业的用户可以自己创造他们自己的field类型

Creating a Schema



[python] view plain copy
from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED
from whoosh.analysis import StemmingAnalyzer

schema = Schema(from_addr=ID(stored=True),
                to_addr=ID(stored=True),
                subject=TEXT(stored=True),
                body=TEXT(analyzer=StemmingAnalyzer()),
                tags=KEYWORD)

如果没有使用一个构造器的关键字参数，可以省略后面的括号，（例如fieldname=TEXT代替fieldname=TEXT()）Whoosh可以为你实例化

你也可以选择使用继承SchemaClass类来创建一个Schema类
[python] view plain copy
from whoosh.fields import SchemaClass, TEXT, KEYWORD, ID, STORED

class MySchema(SchemaClass):
        path = ID(stored=True)
        title = TEXT(stored=True)
        content = TEXT
        tags = KEYWORD
你可以给create_in()或者create_index()函数一个类作为参数而不是他的实例


Modifying the schema after indexing


在你创建index之后，你可以使用add_field()和remove_field()方法来添加或者删除fields。这写方法属于Writer对象
[python] view plain copy
writer = ix.writer()
writer.add_field("fieldname", fields.TEXT(stored=True))
writer.remove_field("content")
writer.commit()
(如果你要使用相同的writer修改schema结构或者向其中添加documents，你必须先调用add_field()或者remove_field()方法)

这些方法Index对象也有，但是当你在Index对象上调用的时候，这些Index对象简单地创建一个writer然后调用相应的方法，然后提交，因此如果需要添加超过一个field，使用writer对象本身更有效率
[python] view plain copy
ix.add_field("fieldname",field.KEYWORD)
在fileddb后端，删除一个field简单地删除schema中的那个schema，索引文件不会变小，那个field里面的数据会保存直到你优化它。优化可以使索引表更紧凑并且移除和已经删除field的相关信息
[python] view plain copy
writer = ix.writer()
writer.add_field("uuid", fields.ID(stored=True))
writer.remove_field("path")
writer.commit(optimize=True)
数据是以field名存储在磁盘文件上面的，不要在优化一个schema之前添加一个和之前删除field相同的field：
[python] view plain copy
writer = ix.writer()
writer.delete_field("path")
# Don't do this!!!
writer.add_field("path", fields.KEYWORD)
（Whoosh将来的版本可能会自动处理这个错误）

Dynamic fields



动态fields可以使用通配符名字将field联系起来
可以使用add()方法（关键字参数glob为真）添加dynamic fields到一个新的schema：
[python] view plain copy
schema = fields.Schema(...)
# Any name ending in "_d" will be treated as a stored
# DATETIME field
schema.add("*_d", fields.DATETIME(stored=True), glob=True)
在一个已经存在的索引上面设置dynamic fields，使用indexWriter.add_field方法就像你添加一个通常的field一样，保证glob参数为True
[python] view plain copy
writer = ix.writer()
writer.add_field("*_d", fields.DATETIME(stored=True), glob=True)
writer.commit()
删除一个dynamic fields可以使用IndexWriter.remove_field()方法（用glob作为名字）
[html] view plain copy
writer = ix.writer()
writer.remove_field("*_d")
writer.commit()
例如。为了使document包含以_id结尾的任意field名字，并且将他与所有的IDfield类型联系起来：
[python] view plain copy
schema = fields.Schema(path=fields.ID)
schema.add("*_id", fields.ID, glob=True)

ix = index.create_in("myindex", schema)

w = ix.writer()
w.add_document(path=u"/a", test_id=u"alfa")
w.add_document(path=u"/b", class_id=u"MyClass")
# ...
w.commit()

qp = qparser.QueryParser("path", schema=schema)
q = qp.parse(u"test_id:alfa")
with ix.searcher() as s:
        results = s.search(）

Advanced schema setup

Field boosts

你可以为一个field指定field boost，这是一个相乘器对于这个fiedl中找到的项，例如，让title field中的项得分两倍于其他域里面的项
schema = Schema(title=TEXT(field_boost=2.0), body=TEXT)

Field types

上面列举的field类型都是fields.FieldType的子类，FieldType是一个相当简单的类，他的属性包含这个field定义的行为
Attribute	Type	Description
format	fields.Format	Defines what kind of information a field recordsabout each term, and how the information is storedon disk.
vector	fields.Format	Optional: if defined, the format in which to storeper-document forward-index information for this field.
scorable	bool	If True, the length of (number of terms in)the field ineach document is stored in the index. Slightly misnamed,since field lengths are not required for all scoring.However, field lengths are required to get properresults from BM25F.
stored	bool	If True, the value of this field is storedin the index.
unique	bool	If True, the value of this field may be used toreplace documents with the same value when the usercallsdocument_update()on an IndexWriter.

预定义的field类型的构造器有允许你自己定制的参数部分，例如：
大多数预定义类型有一个stored关键字参数来设置FieldType.stored
TEXT（）构造器有一个analyzer关键字参数传递给一个格式化对象

Formats

一个Format对象定义field的记录关于每一项包含什么信息以及这写信息如何在磁盘上存储
例如，Existence Format就像这样
Doc
10
20
30

然而Position format可能想这样
Doc	Positions
10	[1,5,23]
20	[45]
30	[7,12]

检索代码传递unicode串，Format对象调用他的analyzer来将串分成token，然后为每个token编码

Class name	Description
Stored	A “null” format for fields that are stored but not indexed.
Existence	Records only whether a term is in a document or not, i.e. itdoes not store term frequency. Useful for identifier fields(e.g. path or id) and “tag”-type fields, where the frequencyis expected to always be 0 or 1.
Frequency	Stores the number of times each term appears in each document.
Positions	Stores the number of times each term appears in each document,and at what positions.
STORED field类型使用Stored format（什么都不做，因此不索引）ID类型使用Existence format，KEYWORD类型使用Frequency format，TEXT使用Position类型如果它以phrase=True的形式被实例化（默认）或者Freqyency format如果phrase=False

另外，下列格式可能为专业用户提供提供某些方便，但是现在Whoosh没有实现

Class name	Description
DocBoosts	Like Existence, but also stores per-document boosts
Characters	Like Positions, but also stores the start and end characterindices of each term
PositionBoosts	Like Positions, but also stores per-position boosts
CharacterBoosts	Like Positions, but also stores the start and end characterindices of each term and per-position boosts

Vectors

主要的索引是一个反向索引。他将document里面出现的项与document建立映射关系。在存储forwad index（向前索引）的时候也可能有用，也称之为term vector，它将document和其中出现的项建立映射关系
例如，假设一个反向索引像这样：
Term	Postings
apple	[(doc=1, freq=2), (doc=2, freq=5), (doc=3, freq=1)]
bear	[(doc=2, freq=7)]
相应的向前索引，或者称之为term vector可能是这样：
Doc	Postings
1	[(text=apple, freq=2)]
2	[(text=apple, freq=5), (text='bear', freq=7)]
3	[(text=apple, freq=1)]



How to Index documents

Creating an index obiect

可以使用index.create_in()函数创建index对象：
[python] view plain copy
import os, os.path
from whoosh import index

#os.path.exists(path) 如果path存在,返回True;如果path不存在,返回False。
if not os.path.exists("indexdir"):
    os.mkdir("indexdir")

ix = index.create_in("indexdir", schema)
带开一个已经存在某个目录的索引，使用index.open_dir()
[python] view plain copy
import whoosh.index as index

ix = index.open_dir("indexdir")
这些是便利方法：
[python] view plain copy
from whoosh.filedb.filestore import FileStorage
storage = FileStorage("indexdir")

# Create an index
ix = storage.create_index(schema)

# Open an existing index
storage.open_index()
你和index对象一起创建的schema对象是可序列化的并且和index一起存储
你可以在同一个目录下面使用多个索引，用关键字参数分开
[python] view plain copy
# Using the convenience functions
ix = index.create_in("indexdir", schema=schema, indexname="usages")
ix = index.open_dir("indexdir", indexname="usages")

# Using the Storage object
ix = storage.create_index(schema, indexname="usages")
ix = storage.open_index(indexname="usages")

Clearing the index

在一个目录上调用index.craete_in()函数可以清除已经存在的索引的内容
可以用函数index.exist_in()来检测制定目录上面是否有一个有效的索引
[html] view plain copy
exists = index.exists_in("indexdir")
usages_exists = index.exists_in("indexdir", indexname="usages")
(你也可以简单地删除目录上面的索引文件，例如如果索引目录只有一个，使用shutil.rmtree()来移除索引然后再重新创建

Indexing documents

一旦你创建了索引对象，你可以使用IndexWriter对象向其中添加document，可以使用Index.writer()来获取writer对象：
[python] view plain copy
ix = index.open_dir("index")
writer = ix.writer()
创建writer的同时会对index加锁，因此同一时间只能有一个线程/进程进行写操作

Note：因为创建writer的时候会创建一个锁，因此在多进程/多线程的时候如果已经有一个writer对象打开那么再打开一个writer的时候可能会抛出一个异常（whoosh.store.LockError）Whoosh有一些例子实现writer的锁操作（whoosh.writing.AsncWriter和whoosh.writing.BufferedWriter）
Note：在writer打开并且提交的过程中，index是可读的，对已经存在reader无影响并且创建的新的reader对象也可以打开index文件，writer提交之后，已经存在的reader对象只能读到没有提交之前的内容，而新创建的reader可以读到最新提交的内容。

IndexWriter对象的add_document(**kwarg)方法接受关键字参数：
[python] view plain copy
writer = ix.writer()
writer.add_document(title=u"My document", content=u"This is my document!",
                    path=u"/a", tags=u"first short", icon=u"/icons/star.png")
writer.add_document(title=u"Second try", content=u"This is the second example.",
                    path=u"/b", tags=u"second short", icon=u"/icons/sheep.png")
writer.add_document(title=u"Third time's the charm", content=u"Examples are many.",
                    path=u"/c", tags=u"short", icon=u"/icons/book.png")
writer.commit()
你不需要为每一个field提供一个值，Whoosh不关心你是否漏掉某个field，被索引的field必须是unicode串，而被存储而不被索引的field可以时任意的可序列化的对象
[python] view plain copy
writer.add_document(path=u"/a", title=u"A", content=u"Hello there")
writer.add_document(path=u"/a", title=u"A", content=u"Deja vu!")
这将添加两个document到索引，见下文的update_document方法，它使用unique field来替换而不是追加内容

Indexing and storing different values for the same field

如果你有一个既需要索引又需要存储的field，你可以索引一个unicode值但是存储一个不同的对象（通常是这样，但是有时候非常有用）使用一个特别的关键字参数_stored_<fieldname>正常的值将被分析和索引，但是存储的这个值将会在结果里面出现：
[python] view plain copy
writer.add_document(title=u"Title to be indexed", _stored_title=u"Stored title")

Finishing adding documents

一个IndexWriter对象就像一个数据库的事物对象。你可以给你的索引用一系列的改变然后一次性的提交。使用commit将IndexWriter提交的内容保存
[python] view plain copy
writer.commit()
一旦你的document在index里面，你就可以搜索他们，如果你想关闭writer而不保存任何提交的东西，使用cancel()
[html] view plain copy
writer.cancel()
记住只要你有一个writer是打开的，没有其它的线程能够另外得到一个writer然后修改index，并且一个writer通常打开着几个文件，因此你在完成工作前必须调用commit或者cancel方法

Merging segements

一个Whoosh filddb的index事实上是一个或者更多个被称为segement的“sub-index”子索引构成的。因此当你添加一个新的document到index里面的时候，
Whoosh将创建一个新的segement到一个已经存在segement里面而不是把他放到documents里面（这样开销很大，因为涉及到磁盘文件上项的排序）
因此当你搜索索引的时候，Whoosh会搜索不同的segement然后将结果整合到一起，看起来就像在同一个index一样（这种设计是模仿Lucene）
因此添加documents的时候有一个segement比不停地重新写入index更为高效。但是搜索多个segemet多少会有点拖慢速度，并且你的segement越多，速度越慢，
因此当你commit的时候Whoosh有一个自己的算法来保证小的segement能够整合成更少的大一点的segement。
如果不想在commit的时候整合segement，可以将关键字参数merge设为False
[python] view plain copy
writer.commit(merge = False)
若想整合所有的segement，优化索引可以使用optimize关键字参数
[python] view plain copy
writer.commit(optimize = True)
因为优化的过程需要重写index上面的所有信息，因此在一个大的索引上面可能会非常慢，通常建议使用Whoosh自己的优化算法而不是一次性地优化

（Index对象也有optimize()方法来优化index它简单地创建一个writer然后调用commit(optimize = True)）

如果你想对整合的过程有更多的控制，一可以重写整合策略的函数然后把他当做commit的参数，具体见NO_MERGE,MERGE_SMALL,OTIMIZE 函数在whoosh.fieldb.filewriting模块中的实现。

Delete Document

你可以用IndexWriter下列方法删除document然后使用commit方法在磁盘上删除
delete_document(document)
使用自己内部的document number删除一个document较低级的方法
is_deleted(docnum)
返回真如果这个给定数字的文档被删除了
delete_by_term(fieldname,termtext)
删除任何包含这个给定的termtext的field
delete_by_query(query)
删除任何匹配这个query的documents

[python] view plain copy
#Delete document by its path--this field must be indexed
ix.delete_by_term("path",u'/a/b/c')
#save the deletion to disk
ix.commit()
在filedb的后端，删除一个document实际上他就是简单地把这个document添加到一个删除列表，当你搜索的时候他不会返回删除列表里面的内容，
但是索引里面的内容仍然存在，他们的统计信息也不会被更新，直到你整合segement（这因为删除信息肯定会立即涉及到重写磁盘，这样效率会很低）

Updating document

如果你想替换掉一个已经存在的document，一可以先删除他然后再添加，当然你也可以使用IndexWriter对象的update_document方法一步完成
对于update_document这个方法，你必须确保至少有一个field的内容是唯一的，Whoosh会使用这个内容来搜索结果然后删除：

[python] view plain copy
from whoosh.fields import Schema,ID,TEXT

schema = Schema (path=ID (unique=True),content=TEXT)

ix = index.create_in("index")
writer = ix.writer()
writer.add_document(path=u"/a",content=u"the first document")
writer.add_cocument(path=u"/b",content=u"The second document")
writer.commit()

writer = ix.writer()
#Because "path" is marked as unique,calling update_document with path = u"/a"
#will delete any existing document where the path field contains /a
writer.update_document(path=u"/a",content="Replacement for the first document")
writer.commit()
这个“unique”field必须是被索引的
如果没有document能够搭配这个unique的field，那么这个方法就跟add_document一样

“unique”的field和update_document方法仅仅是为了方便删除和添加使用，Whoosh没有固有的唯一性概念，你也没法在使用add_document的时候强制指定唯一性

Incremental Indexing(增量索引)
当你索引一个documents的时候，可能有两种方式：一个是给所有的内容全部索引，一个时只更新发生改变的内容
给所有内容全部索引是很简单的：
[python] view plain copy
import os.path
from whoosh import index
from whoosh.fields import Schema, ID, TEXT

def clean_index(dirname):
  # Always create the index from scratch
  ix = index.create_in(dirname, schema=get_schema())
  writer = ix.writer()

  # Assume we have a function that gathers the filenames of the
  # documents to be indexed
  for path in my_docs():
  add_doc())writer, path)

  writer.commit()


def get_schem()
    return Schema(path=ID(unique=True, stored=True), content=TEXT)


def add_doc(writer, path):
    fileobj=open(path, "rb")
    content=fileobj.read()
    fileobj.close()
    writer.add_document(path=path, content=content)
对于一个很小的document，每次都全部索引可能会非常快，但是对于大的documents，你可能需要仅仅更新改变的内容
为了这么做我们需要存储每一个document的最后修改时间，在这个例子里我们使用mtime来简化：
[python] view plain copy
def get_schema()
return Schema())path=ID(unique=True, stored=True), time=STORED, content=TEXT)

def add_doc(writer, path):
fileobj=open())path, "rb")
content=fileobj.read()")
fileobj.close()")
modtime = os.path.getmtime())path)
writer.add_document()
path=path, content=content, time=modtime)
现在我们能够判断是清除还是增量索引：
[python] view plain copy
def index_my_docs(dirname, clean=False):
  if clean:
    clean_index(dirname)
  else:
    incremental_index(dirname)


def incremental_index(dirname)
    ix = index.open_dir(dirname)

    # The set of all paths in the index
    indexed_paths = set()
    # The set of all paths we need to re-index
    to_index = set()

    with ix.searcher() as searcher:
      writer = ix.writer()

      # Loop over the stored fields in the index
      for fields in searcher.all_stored_fields():
        indexed_path = fields['path']
        indexed_paths.add(indexed_path)

        if not os.path.exists(indexed_path):
          # This file was deleted since it was indexed
          writer.delete_by_term('path', indexed_path)

        else:
          # Check if this file was changed since it
          # was indexed
          indexed_time = fields['time']
          mtime = os.path.getmtime(indexed_path)
          if mtime > indexed_time:
            # The file has changed, delete it and add it to the list of
            # files to reindex
            writer.delete_by_term('path', indexed_path)
            to_index.add(indexed_path)

      # Loop over the files in the filesystem
      # Assume we have a function that gathers the filenames of the
      # documents to be indexed
      for path in my_docs():
        if path in to_index or path not in indexed_paths:
          # This is either a file that's changed, or a new file
          # that wasn't indexed before. So index it!
          add_doc(writer, path)

      writer.commit()
增量索引功能：


1.在所有已经索引的路径里面循环
a.如果文件不存在了，在现有的document里面删除
b.如果文件仍然存在但是被修改过，添加到需要修改的列表
c.如果文件存在不管是否修改过，添加到一索引的路径里面

2.在磁盘上的所有文件遍历
a.如果文件不是以索引路径集合的一部分，那么这个文件是新的，需要索引他
b.如果路径是修改列表的一部分，也需要更新他
c.否则，跳过