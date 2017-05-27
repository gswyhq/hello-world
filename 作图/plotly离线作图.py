#!/usr/bin/python3
# coding: utf-8

# 在执行脚本时，它将打开一个Web浏览器，并绘制绘图。
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.graph_objs import Scatter, Figure, Layout

plot([Scatter(x=[1, 2, 3], y=[3, 1, 6])])

# 在Jupyter Notebook环境中离线绘制图形。首先，您需要启动Plotly Notebook模式
init_notebook_mode(connected=True)

iplot([{"x": [1, 2, 3], "y": [3, 1, 6]}])

from plotly.graph_objs import *
import numpy as np

x = np.random.randn(2000)
y = np.random.randn(2000)
iplot([Histogram2dContour(x=x, y=y, contours=Contours(coloring='heatmap')),
       Scatter(x=x, y=y, mode='markers', marker=Marker(color='white', size=3, opacity=0.3))], show_link=False)

from plotly.graph_objs import *
import pandas as pd

df = pd.read_csv('https://plot.ly/~etpinard/191.csv')

iplot({
    'data': [
        Scatter(x=df['"{}{}"'.format(continent,', x')],
                y=df['"{}{}"'.format(continent,', y')],
                text=df['"{}{}"'.format(continent,', text')],
                marker=Marker(size=df['"{}{}"'.format(continent,', size')], sizemode='area', sizeref=131868,),
                mode='markers',
                name=continent) for continent in ['Africa', 'Americas', 'Asia', 'Europe', 'Oceania']
    ],
    'layout': Layout(xaxis=XAxis(title='Life Expectancy'), yaxis=YAxis(title='GDP per Capita', type='log'))
}, show_link=False)

import cufflinks as cf
iplot(cf.datagen.lines().iplot(asFigure=True,
                               kind='scatter',xTitle='Dates',yTitle='Returns',title='Returns'))

import plotly.plotly as py

fig = py.get_figure('https://plot.ly/~jackp/8715', raw=True)
iplot(fig)


import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()

offline.iplot({'data': [{'y': [4, 2, 3, 4]}],
               'layout': {'title': 'Test Plot',
                          'font': dict(size=16)}},
             image='png')

import plotly.offline as offline
import plotly.graph_objs as go

offline.plot({'data': [{'y': [4, 2, 3, 4]}],
               'layout': {'title': 'Test Plot',
                          'font': dict(size=16)}},
             image='png')

mport plotly
help(plotly.offline.iplot)


