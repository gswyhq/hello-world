#!/usr/bin/python3
# coding: utf-8

import os
import sys
from bs4 import BeautifulSoup
import json

def html2ipynb(sourceHtml, targetIpynb):

    dictionary = {'nbformat': 4, 'nbformat_minor': 1, 'cells': [], 'metadata': {}}

    # print("html2ipynb")
    # print("Source (html)  : '%s'" % sourceHtml)
    # print("Target (ipynb) : '%s'" % targetIpynb)

    response = open(sourceHtml, encoding='utf-8')
    text = response.read()

    soup = BeautifulSoup(text, 'lxml')

    for d in soup.findAll("div"):
        if 'class' in d.attrs.keys():
            for clas in d.attrs["class"]:
                if clas in ["text_cell_render", "input_area"]:
                    # code cell
                    if clas == "input_area":
                        cell = {}
                        cell['metadata'] = {}
                        cell['outputs'] = []
                        # this removes all empty lines
                        # new_source = "\n".join([s for s in d.get_text().splitlines() if s.strip()])
                        new_source = d.get_text().splitlines()
                        # remove first line if empty
                        if not new_source[0].strip():
                            new_source = new_source[1:]
                        # remove last lines if empty
                        if not new_source[-1].strip():
                            new_source = new_source[0:-1]
                        cell['source'] = "\n".join(new_source)
                        cell['execution_count'] = None
                        cell['cell_type'] = 'code'
                        dictionary['cells'].append(cell)

                    else:
                        cell = {}
                        cell['metadata'] = {}

                        cell['source'] = [d.decode_contents()]
                        cell['cell_type'] = 'markdown'
                        dictionary['cells'].append(cell)

    open(targetIpynb, 'w').write(json.dumps(dictionary))
    response.close()
    print("成功将文件`{}`转换为：`{}`".format(sourceHtml, targetIpynb))
    
def main():
    error_msg = '使用方法示例： python3 html2ipynb.py code.html '
    try:
        assert len(sys.argv) > 1, error_msg
        sourceHtml = sys.argv[1]
        assert sourceHtml.lower()[-5:] == '.html', error_msg
        targetIpynb = sourceHtml[:-5] + '.ipynb'
        assert not os.path.isfile(targetIpynb), '文件`{}`已经存在，请先移除该文件'.format(targetIpynb)
        html2ipynb(sourceHtml, targetIpynb)
    except AssertionError as e:
        print(e)
    except Exception as e:
        print(e)
        print(error_msg)

if __name__ == '__main__':
    main()

# HTML文件转换为ipynb文件；
# 来源：https://raw.githubusercontent.com/sgomezvillamor/html2ipynb/master/html2ipynb.ipynb
# $ python3 html2ipynb.py /home/gswyhq/Downloads/Code.html
# 成功将文件`/home/gswyhq/Downloads/Code.html`转换为：`/home/gswyhq/Downloads/Code.ipynb`