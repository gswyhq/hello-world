
def setup_color(y1_list, y0_list):
    writer = pd.ExcelWriter(r'setup_color.xlsx', engine='xlsxwriter')
    df = pd.read_excel(writer)
    df.to_excel(writer, index=False, sheet_name='sheet')
    workbook = writer.book

    # 设置excel标题行，指定单元格的背景色
    header_fmt = workbook.add_format({
        # 'font_size': 14,
        # 'align': 'left', # 水平对齐方式
        # 'valign': 'vcenter', # 垂直对齐方式 
        # 'bold': True, # 是否字体加粗
        'fg_color': '#D7E4BC', # 单元格背景颜色
        # 'border': 1, # 单元格边框宽度
        # 'text_wrap': True,  # 是否自动换行
    })
    sheet_table = writer.sheets['sheet']
    for col_num, value in enumerate(df.columns.values):
        if value in ['投诉类', '风险类', '意图标签', '咨询回访']:
            sheet_table.write(0, col_num, value, header_fmt)
        else:
            sheet_table.write(0, col_num, value)

    yellow_format = workbook.add_format({'fg_color': '#FFEE99'})
    red_format = workbook.add_format({'fg_color': 'red'})
    # uid列，若在列表 y1_list中，则设置背景色为红色，若在y0_list中，则设置背景色为黄色
    for row, value in enumerate(df['uid'].values, 1):
        if value in y1_list:
            sheet_table.write(row, 1, value, red_format)
        if value in y0_list:
            sheet_table.write(row, 1, value, yellow_format)

    writer.save()
    writer.close()

# 更多示例，参见：https://xlsxwriter.readthedocs.io/chart_examples.html#chart-examples
# https://xlsxwriter.readthedocs.io/working_with_conditional_formats.html?highlight=conditional_format

