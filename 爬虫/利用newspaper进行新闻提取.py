


https://github.com/codelucas/newspaper

需要进行如下安装：
$ sudo apt-get install libxml2-dev libxslt-dev
$ sudo apt-get install libjpeg-dev zlib1g-dev libpng12-dev
$ curl https://raw.githubusercontent.com/codelucas/newspaper/master/download_corpora.py | python3
$ pip3 install newspaper3k

>>> import newspaper
>>> sina_paper = newspaper.build('http://www.sina.com.cn/', language='zh')

>>> for category in sina_paper.category_urls():
>>>     print(category)
http://health.sina.com.cn
http://eladies.sina.com.cn
http://english.sina.com
...

>>> article = sina_paper.articles[0]
>>> article.download()
>>> article.parse()

>>> print(article.text)
新浪武汉汽车综合 随着汽车市场的日趋成熟，
传统的“集全家之力抱得爱车归”的全额购车模式已然过时，
另一种轻松的新兴 车模式――金融购车正逐步成为时下消费者购
买爱车最为时尚的消费理念，他们认为，这种新颖的购车
模式既能在短期内
...

>>> print(article.title)
两年双免0手续0利率 科鲁兹掀背金融轻松购_武汉车市_武汉汽
车网_新浪汽车_新浪网
