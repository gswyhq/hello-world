
# 多个列表求交集：
from functools import reduce
reduce(lambda x,y: x&y, [set(t) for t in [['Baoxianchanpin', 'Jingyinglinghang', 'Yingxiao', 'Chengxin_yyb_228'],
 ['Baoxianchanpin', 'Zhiyingweilai', 'Yinbao', 'Chengxin_yyb_228'],
 ['Baoxianchanpin', 'Nuanbaobao', 'Yingxiao', 'Chengxin_yyb_228'],
 ['Baoxianchanpin', 'Nuanbaobao', 'Yonghu', 'Chengxin_yyb_228'],
 ['Baoxianchanpin', 'Yiwaishanghaibkuan', 'Yingxiao', 'Chengxin_yyb_228'],
 ['Baoxianchanpin', 'Yiwaishanghaibkuan', 'Yinbao', 'Chengxin_yyb_228']]])
Out[36]: {'Baoxianchanpin', 'Chengxin_yyb_228'}

