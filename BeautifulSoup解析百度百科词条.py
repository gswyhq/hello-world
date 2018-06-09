#encode:utf-8
#! usr/bin/python3.5
"""解析百度百科词条"""
import os,re
import time
from bs4 import BeautifulSoup



def read_html(f='/media/gswewf/000724FA000F917A/Baike/part-r-00001',split=0):
    """读取文件，并分割,或者解析"""
    txt=''
    textall=''
    text_hrefall=''
    count=0
    with open(f,encoding='utf-8') as fi:
        line=fi.readline()
        while(line):
            m=re.search(r"http:\/\/baike\.baidu\.com\/view\/(\d+)\.htm\*\*\*\*\*<!DOCTYPE html>",line)
            if m:
                count+=1
                if count%1000==0:
                    print("已解析文件数：",count)
                    write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/纯文本", os.path.split(f)[-1]),textall)
                    write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/包括链接等", os.path.split(f)[-1]),text_hrefall)
                    textall=''
                    text_hrefall=''
                if txt:
                    if split:
                        write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/split_file",file),txt)
                    else:
                        try:
                            text,text_href=parser_html(txt)
                            split_line='\n'+"*"*10+file+"*"*10+'\n'#分割线
                            textall+=split_line+text
                            text_hrefall+=split_line+text_href
                            #write_txt(os.path.join(r"E:\Baike\纯文本",file),text)
                            #write_txt(os.path.join(r"E:\Baike\包括链接等",file),text_href)
                        except  Exception as e :
                            print("解析出错：",file,e)
                            write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/split_file",file),txt)
                #file=os.path.join("E:\Baike\split_file",m.group(1))
                file=m.group(1)
                txt=''
            txt+=line
            line=fi.readline()

        if split:
            write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/split_file",file),txt)
        else:
            try:
                text,text_href=parser_html(txt)
                split_line='\n'+"*"*10+file+"*"*10+'\n'#分割线
                textall+=split_line+text
                text_hrefall+=split_line+text_href
                write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/纯文本", os.path.split(f)[-1]),textall)
                write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/包括链接等", os.path.split(f)[-1]),text_hrefall)
                #write_txt(os.path.join(r"E:\Baike\纯文本",file),text)
                #write_txt(os.path.join(r"E:\Baike\包括链接等",file),text_href)
            except  Exception as e :
                print("解析出错：",file,e)
                write_txt(os.path.join(r"/media/gswewf/000724FA000F917A/Baike/split_file",file),txt)
            
        

def get_html(path=r"/media/gswewf/000724FA000F917A/Baike/Baike"):
    """根据路径，提取路径下所有文件内的HTML内容。"""
    for root, dirs, fs in os.walk(path):
        files=[]
        for file in fs:
            if '7z' not in file:
                files.append(os.path.join(root,file))
        
#files=['0010','10','0153','0472','0571','1000','006236']
#files=['0010']
    """
    html_doc=''
    for f in files:
        file=os.path.join(path,f)
        with open(file,encoding='utf-8') as f_read:
            print(file)
            html_doc+=f_read.read()
    """
    return files

def write_txt(file,txt):
    """将文本写入文件"""
    with open(file,'a+',encoding='utf-8') as f2:
        f2.write(txt)
            
def parser_html(html_doc):
    """解析HTML文件，提取文本内容，或提取包括链接的文本内容。"""
    soup = BeautifulSoup(html_doc,'lxml')

    #提取标题
    title=soup.find('h1',class_='title')
    text=title.text
    text_href=text

    #处理多义
    try:
        polysemant=soup.find('div',class_="b-ent-toc slv-bdr b-clearfix b-rlt low-lh  ")
        #>>> polysemant=soup.find('div',class_=["polysemant-list","b-ent-toc slv-bdr b-clearfix b-rlt low-lh  "])
        span=polysemant.find('span',class_="small thin")
        text+='\n'+span.text
        text_href+='\n'+span
        lis=polysemant.find_all('li')
        for li in lis:
            text+='\n'+li.text
            text_href+='\n'+li
            #polysemant=soup.find('a',target="_blank")
    except:
        pass

    #提取正文内容
    try:
        main_content=soup.find('div',class_="text lazyload")
        txts=main_content.find_all('div',class_='para')


    except AttributeError:
        main_content=soup.find('div',class_=["text","content-bd main-body"])
        txts=main_content.find_all('div',class_='para')

    for t in txts:
        text+='\n'+t.text#若要保留超链接等，就仅仅需要t即可，而不用.text
        text_href+='\n'+str(t)

    #提取开放分类、标签、参考分类等
    open_title=soup.find_all('dl')
    for child in open_title:
        for t in child.find_all(['dt','dd']):
            text+='\n'+t.text#若要保留超链接等，就仅仅需要t即可，而不用.text
            text_href+='\n'+str(t)

    #print(text_href[-200:])

    return text,text_href



if __name__ == '__main__':
    start=time.time()
    #files=get_html()[:5]

    files=['/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00039',
 '/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00040',
 '/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00041',
 '/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00042',
 '/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00043',
 '/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00062',
 '/media/gswewf/000724FA000F917A/Baike/Baike/part-r-00063']



    for f in files:
        read_html(f)
    print("OK!",time.time()-start)

'''
    
file=r"E:\Baike\split_file\0010"
>>> with open(file,encoding='utf-8')as f:
	html=f.read()

	
>>> soup = BeautifulSoup(html)

main_content=soup.find_all('div',class_="text lazyload")

>>> main_content=soup.find('div',class_="text lazyload")

>>> main_content.text


>>> title=main_content.find('h1',class_='title')
>>> title.string
'红色食品'
>>> txt=main_content.find_all('div',class_='para')
>>> txt[0].string
'红色食品是指食品为红色、橙红色或棕红色的食品。科学家认为，多吃些红色食品可预防感冒。红色食品有红柿椒、西红柿、胡萝卜、红心白薯、红果（山楂）、红苹果、草莓、红枣、老南瓜、红米、柿子等。 有治疗缺铁性贫血和缓解疲劳的作用，对乳腺癌等肿瘤疾病有防治作用，给人以兴奋感，有增加食欲，光洁皮肤，增强表皮细胞再生和防止皮肤衰老，预防感冒等作用。'
>>> txt[1].string
>>> txt[0].text
'红色食品是指食品为红色、橙红色或棕红色的食品。科学家认为，多吃些红色食品可预防感冒。红色食品有红柿椒、西红柿、胡萝卜、红心白薯、红果（山楂）、红苹果、草莓、红枣、老南瓜、红米、柿子等。 有治疗缺铁性贫血和缓解疲劳的作用，对乳腺癌等肿瘤疾病有防治作用，给人以兴奋感，有增加食欲，光洁皮肤，增强表皮细胞再生和防止皮肤衰老，预防感冒等作用。


'''
