#!/usr/bin/python3

'''
<html><head lang="en">
    <meta charset="UTF-8">
    <title></title>
    <link rel="stylesheet" href="//cdn.datatables.net/1.10.2/css/jquery.dataTables.min.css">
    <script type="text/javascript" src="//cdnjs.cloudflare.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
    <script type="text/javascript" src="//cdn.datatables.net/1.10.2/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" src="../static/js/excel2html.js"></script>
</head>
<body>
    <div id="table_test_wrapper" class="dataTables_wrapper no-footer"><div id="table_test_filter" class="dataTables_filter"><label>Search:<input type="search" class="" placeholder="" aria-controls="table_test"></label></div><div id="table_test_processing" class="dataTables_processing" style="display: none;">Processing...</div><table id="table_test" class="display dataTable no-footer" style="width: 1904px;" role="grid" aria-describedby="table_test_info">
            <thead>
                <tr role="row"><th class="sorting_asc" tabindex="0" aria-controls="table_test" rowspan="1" colspan="1" aria-label="ID: activate to sort column ascending" aria-sort="ascending" style="width: 445px;">ID</th><th class="sorting" tabindex="0" aria-controls="table_test" rowspan="1" colspan="1" aria-label="Name: activate to sort column ascending" style="width: 533px;">Name</th><th class="sorting" tabindex="0" aria-controls="table_test" rowspan="1" colspan="1" aria-label="Ability: activate to sort column ascending" style="width: 446px;">Ability</th><th class="sorting" tabindex="0" aria-controls="table_test" rowspan="1" colspan="1" aria-label="Score: activate to sort column ascending" style="width: 336px;">Score</th></tr>
            </thead>
    <tbody><tr role="row" class="odd"><td class="sorting_1">20111005.0</td><td>steve jobs</td><td>creative</td><td>90.0</td></tr><tr role="row" class="even"><td class="sorting_1">20170120.0</td><td>donald trump</td><td>self-control</td><td>10.0</td></tr><tr role="row" class="odd"><td class="sorting_1">20180102.0</td><td>bill gate</td><td>business</td><td>80.0</td></tr><tr role="row" class="even"><td class="sorting_1">20180103.0</td><td>tim cook</td><td>creative</td><td>50.0</td></tr></tbody></table><div class="dataTables_info" id="table_test_info" role="status" aria-live="polite">Showing 1 to 4 of 4 entries</div></div>
</body></html>

'''

import pandas
with open('C:\Users\zhaoyingh\Desktop\\a.html','r') as f:
    df = pandas.read_html(f.read().decode("gb2312").encode('utf-8'),encoding='utf-8')
print df[0]
bb = pandas.ExcelWriter('out.xlsx')
df[0].to_excel(bb)
bb.close()


import pandas as pd
def convert_to_html(result,title):
    d = {}
    index = 0
    for t in title:
        d[] = result[index]
        index +=1
    df = pd.DataFrame(d)
    #如数据过长，可能在表格中无法显示，加上pd.set_option语句可以避免这一情况
    pd.set_option('max_colwidth',200)
    df = df [title]
    h =df.to_html(index=False)
    return h

