#!/usr/bin/python3
# coding: utf-8

# 第一步：下载数据、解压、导入mysql
# gswyhq@gswyhq-pc:~$ wget http://downloads.mysql.com/docs/world.sql.zip
# gswyhq@gswyhq-pc:~$ unzip world.sql.zip
# Navicat for MySQL: 选择一个数据表；右键 -> Execute SQL File -> 选择刚解压的数据即可

# 或者：
# 命令行连接mysql
# shell> sudo mysql -uroot -p

# 执行mysql命令，导入数据
# mysql> CREATE DATABASE world;
# mysql> USE world;
# mysql> SOURCE /home/ubuntu/world.sql;

import pymysql
import pandas as pd

conn = pymysql.connect(host="localhost", user="root", password="123456", db="world")
with conn.cursor() as cursor:
    cursor.execute('select Name, Continent, Population, LifeExpectancy, GNP from country')
    rows = cursor.fetchall()
    print(rows)


df = pd.DataFrame( [[ij for ij in i] for i in rows] )
df.rename(columns={0: 'Name', 1: 'Continent', 2: 'Population', 3: 'LifeExpectancy', 4:'GNP'}, inplace=True)
df = df.sort(['LifeExpectancy'], ascending=[1])

import plotly.plotly as py
from plotly.graph_objs import Scatter, Layout, XAxis, YAxis, Data, Figure, Marker, Line
# 设置登陆信息及api秘钥
py.sign_in(username="gswyhq", api_key='VRsmiPYfSNqJ3rEWk6JX')
# 此处的用户名是在网站“https://plot.ly/”注册后的用户名；api_key并不是注册时候的登陆密码，
# 而是在页面“https://plot.ly/settings/api”点击生成秘钥按钮“Regenerate Key”后，生成的秘钥；
# 生成秘钥后，可以在本地文件中记录秘钥：
# gswyhq@gswyhq-pc:~$ vim .plotly/.credentials

# 生成的可视化图像，并不是在本地；而是在页面： https://plot.ly/~gswyhq/0


country_names = df['Name']
# for i in range(len(country_names)):
#     try:
#         country_names[i] = str(country_names[i]).decode('utf-8')
#     except:
#         country_names[i] = 'Country name decode error'
def fig1():
    trace1 = Scatter(
            x=df['LifeExpectancy'],
            y=df['GNP'],
            text=country_names,
            mode='markers'
    )
    layout = Layout(
            xaxis=XAxis(title='Life Expectancy'),
            yaxis=YAxis(type='log', title='GNP')
    )
    data = Data([trace1])
    fig = Figure(data=data, layout=layout)
    py.iplot(fig, filename='world GNP vs life expectancy')

def fig2():
    # (!) Set 'size' values to be proportional to rendered area,
    #     instead of diameter. This makes the range of bubble sizes smaller
    sizemode = 'area'

    # (!) Set a reference for 'size' values (i.e. a population-to-pixel scaling).
    #     Here the max bubble area will be on the order of 100 pixels
    sizeref = df['Population'].max() / 1e2 ** 2

    colors = {
        'Asia': "rgb(255,65,54)",
        'Europe': "rgb(133,20,75)",
        'Africa': "rgb(0,116,217)",
        'North America': "rgb(255,133,27)",
        'South America': "rgb(23,190,207)",
        'Antarctica': "rgb(61,153,112)",
        'Oceania': "rgb(255,220,0)",
    }

    # Define a hover-text generating function (returns a list of strings)
    def make_text(X):
        return 'Country: %s\
        <br>Life Expectancy: %s years\
        <br>Population: %s million' \
               % (X['Name'], X['LifeExpectancy'], X['Population'] / 1e6)

        # Define a trace-generating function (returns a Scatter object)

    def make_trace(X, continent, sizes, color):
        return Scatter(
                x=X['GNP'],  # GDP on the x-xaxis
                y=X['LifeExpectancy'],  # life Exp on th y-axis
                name=continent,  # label continent names on hover
                mode='markers',  # (!) point markers only on this plot
                text=X.apply(make_text, axis=1).tolist(),
                marker=Marker(
                        color=color,  # marker color
                        size=sizes,  # (!) marker sizes (sizes is a list)
                        sizeref=sizeref,  # link sizeref
                        sizemode=sizemode,  # link sizemode
                        opacity=0.6,  # (!) partly transparent markers
                        line=Line(width=3, color="white")  # marker borders
                )
        )

    # Initialize data object
    data = Data()

    # Group data frame by continent sub-dataframe (named X),
    #   make one trace object per continent and append to data object
    for continent, X in df.groupby('Continent'):
        sizes = X['Population']  # get population array
        color = colors[continent]  # get bubble color

        data.append(
                make_trace(X, continent, sizes, color)  # append trace to data object
        )

        # Set plot and axis titles
    title = "Life expectancy vs GNP from MySQL world database (bubble chart)"
    x_title = "Gross National Product"
    y_title = "Life Expectancy [in years]"

    # Define a dictionary of axis style options
    axis_style = dict(
            type='log',
            zeroline=False,  # remove thick zero line
            gridcolor='#FFFFFF',  # white grid lines
            ticks='outside',  # draw ticks outside axes
            ticklen=8,  # tick length
            tickwidth=1.5  # and width
    )

    # Make layout object
    layout = Layout(
            title=title,  # set plot title
            plot_bgcolor='#EFECEA',  # set plot color to grey
            hovermode="closest",
            xaxis=XAxis(
                    axis_style,  # add axis style dictionary
                    title=x_title,  # x-axis title
                    range=[2.0, 7.2],  # log of min and max x limits
            ),
            yaxis=YAxis(
                    axis_style,  # add axis style dictionary
                    title=y_title,  # y-axis title
            )
    )

    # Make Figure object
    fig = Figure(data=data, layout=layout)

    # (@) Send to Plotly and show in notebook
    py.iplot(fig, filename='s3_life-gdp')

def main():
    fig2()


if __name__ == '__main__':
    main()
