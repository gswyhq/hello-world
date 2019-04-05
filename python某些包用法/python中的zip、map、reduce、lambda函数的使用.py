
# 多个列表求交集：
from functools import reduce
reduce(lambda x,y: x&y, [set(t) for t in [['Baoxianchanpin', 'Jingyinglinghang', 'Yingxiao', 'Xincheng_yyb_228'],
 ['Baoxianchanpin', 'Zhiyingweilai', 'Yinbao', 'Xincheng_yyb_228'],
 ['Baoxianchanpin', 'Nuanbaobao', 'Yingxiao', 'Xincheng_yyb_228'],
 ['Baoxianchanpin', 'Nuanbaobao', 'Yonghu', 'Xincheng_yyb_228'],
 ['Baoxianchanpin', 'Yiwaishanghaibkuan', 'Yingxiao', 'Xincheng_yyb_228'],
 ['Baoxianchanpin', 'Yiwaishanghaibkuan', 'Yinbao', 'Xincheng_yyb_228']]])
Out[36]: {'Baoxianchanpin', 'Xincheng_yyb_228'}

