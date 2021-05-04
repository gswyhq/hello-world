#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pyecharts.charts import EffectScatter

from pyecharts import options as opts

es =EffectScatter()
es._xaxis_data=[21]
es.add_yaxis("", [ 10], symbol_size=20, effect_opts = opts.EffectOpts(scale=3.5, period=3), symbol="pin")
es._xaxis_data=[12]
es.add_yaxis("", [ 20], symbol_size=12, effect_opts = opts.EffectOpts(scale=4.5, period=4), symbol="rect")
es._xaxis_data=[34]
es.add_yaxis("", [ 30], symbol_size=30, effect_opts = opts.EffectOpts(scale=5.5, period=5),symbol="roundRect")
es._xaxis_data=[32]
es.add_yaxis("", [ 40], symbol_size=10, effect_opts = opts.EffectOpts(scale=6.5, brush_type='fill'),symbol="diamond")
es._xaxis_data=[45]
es.add_yaxis("", [ 50], symbol_size=16, effect_opts = opts.EffectOpts(scale=5.5, period=3),symbol="arrow")
es._xaxis_data=[65]
es.add_yaxis("", [ 61], symbol_size=6, effect_opts = opts.EffectOpts(scale=2.5, period=3),symbol="triangle")
es.set_series_opts(label_opts = opts.LabelOpts(is_show=True, position='left',formatter="{c}"))
es.render(r'D:\Users\data\card\积分\带有涟漪特效动画的散点图.html')

def main():
    pass


if __name__ == '__main__':
    main()