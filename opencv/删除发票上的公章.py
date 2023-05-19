

# 在OCR识别发票的时候，有时候是不需要识别公章内容，这个时候，可以先把公章去掉：
#去章处理方法
def remove_stamp(invoice_file_name):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    B_channel,G_channel,R_channel=cv2.split(img)     # 注意cv2.split()返回通道顺序
    _,RedThresh = cv2.threshold(R_channel,170,355,cv2.THRESH_BINARY)
    cv2.imwrite('./test/RedThresh_{}.jpg'.format(invoice_file_name),RedThresh)

# 资料来源：https://github.com/guanshuicheng/invoice/app.py
