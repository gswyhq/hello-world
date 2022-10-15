#! /usr/lib/python3
# -*- coding: utf-8 -*-

import xmltojson,json


def main():
    print(json.dumps(xmltodict.parse("""
... <mydocument has="an attribute">
... <and>
... <many>elements</many>
... <many>more elements</many>
... </and>
... <plus a="complex">
... element as well
... </plus>
... </mydocument>
... """), indent=4))

    xml = """
    <root xmlns="http://defaultns.com/" <br="">... xmlns:a="http://a.com/"
    xmlns:b="http://b.com/">
    <x>1</x>
    <a:y>2</a:y>
    <b:z>3</b:z>
    </root>
    """

if __name__ == "__main__":
    main()


'''
In[17]: xml="""
...  <mydocument has="an attribute">
...    <and>
...      <many>elements</many>
...      <many>more elements</many>
...    </and>
...    <plus a="complex">
...      element as well
...    </plus>
...  </mydocument>
...  """
In[18]: xmltodict.parse(xml)
Out[18]:
OrderedDict([('mydocument',
              OrderedDict([('@has', 'an attribute'),
                           ('and',
                            OrderedDict([('many',
                                          ['elements', 'more elements'])])),
                           ('plus',
                            OrderedDict([('@a', 'complex'),
                                         ('#text', 'element as well')]))]))])
In[19]: print(json.dumps(xmltodict.parse(xml),indent=4))
{
    "mydocument": {
        "@has": "an attribute",
        "and": {
            "many": [
                "elements",
                "more elements"
            ]
        },
        "plus": {
            "@a": "complex",
            "#text": "element as well"
        }
    }
}
In[156]: xmltodict.parse(xml,xml_attribs=0)
Out[154]:
OrderedDict([('mydocument',
              OrderedDict([('and',
                            OrderedDict([('many',
                                          ['elements', 'more elements'])])),
                           ('plus', 'element as well')]))])


In[20]: xml = """
... <root xmlns="http://defaultns.com/"
...       xmlns:a="http://a.com/"
...       xmlns:b="http://b.com/">
...   <x>1</x>
...   <a:y>2</a:y>
...   <b:z>3</b:z>
... </root>
... """
In[21]: xmltodict.parse(xml, process_namespaces=True) == {
...     'http://defaultns.com/:root': {
...         'http://defaultns.com/:x': '1',
...         'http://a.com/:y': '2',
...         'http://b.com/:z': '3',
...     }
... }
Out[21]: True
In[22]: namespaces = {
...     'http://defaultns.com/': None, # skip this namespace
...     'http://a.com/': 'ns_a', # collapse "http://a.com/" -> "ns_a"
... }
In[23]: xmltodict.parse(xml, process_namespaces=True, namespaces=namespaces) == {
...     'root': {
...         'x': '1',
...         'ns_a:y': '2',
...         'http://b.com/:z': '3',
...     },
... }
Out[23]: True

In[26]: mydict = {
...     'response': {
...             'status': 'good',
...             'last_updated': '2014-02-16T23:10:12Z',
...     }
... }

In[30]: print (xmltodict.unparse(mydict, pretty=True))
<?xml version="1.0" encoding="utf-8"?>
<response>
	<last_updated>2014-02-16T23:10:12Z</last_updated>
	<status>good</status>
</response>
In[31]: mydict = {
...     'text': {
...         '@color':'red',
...         '@stroke':'2',
...         '#text':'This is a test'
...     }
... }
In[32]: print (xmltodict.unparse(mydict, pretty=True))
<?xml version="1.0" encoding="utf-8"?>
<text color="red" stroke="2">This is a test</text>

In[137]: with open('/home/gswewf/gow69/data/美团网城市列表.xml', "rb") as f:
    d = xmltodict.parse(f, xml_attribs=True)
In[138]: d.keys()
Out[136]: odict_keys(['response'])

In[141]: xml = """
        <servers>
          <server>
            <name>server1</name>
            <os>os1</os>
          </server>
        </servers>
        """
In[142]: xmltodict.parse(xml, force_list=('server',))
Out[140]:
OrderedDict([('servers',
              OrderedDict([('server',
                            [OrderedDict([('name', 'server1'),
                                          ('os', 'os1')])])]))])
In[143]: xmltodict.parse(xml)
Out[141]:
OrderedDict([('servers',
              OrderedDict([('server',
                            OrderedDict([('name', 'server1'),
                                         ('os', 'os1')]))]))])



>>> def handle_artist(_, artist):
...     print artist['name']
...     return True
>>>
from gzip import GzipFile
>>> xmltodict.parse(GzipFile('/home/gswewf/gow69/data/zhwiki-20160407-stub-articles1.xml.gz'),
...     item_depth=2, item_callback=handle_artist)


In[158]: def postprocessor(path, key, value):
            try:
                return key + ':int', int(value)
            except (ValueError, TypeError):
                return key, value

In[160]: xmltodict.parse('<a><b>1</b><b>2</b><b>x</b></a>',
            postprocessor=postprocessor)
Out[158]: OrderedDict([('a', OrderedDict([('b:int', [1, 2]), ('b', 'x')]))])
'''



