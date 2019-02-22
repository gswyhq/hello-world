
# 解决方案：
# 1. 生成html table代码
# 2. chrome 屏幕大小调整后截屏
from selenium import webdriver
import numpy as np
import pandas as pd

pda = pd.DataFrame(np.random.random((3,3)))
pda.to_html('/home/gswyhq/Downloads/product23232.html')

# 截屏代码： 
driver = webdriver.Chrome()
driver.set_window_size(1000, 680)
driver.get('file:///C:/Users/KC10/Desktop/data%20clearn/table.html')
driver.save_screenshot('table.png')
driver.quit()


# 方案二： 选择的是matplotlib.pyplot.table

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
  
  
df = pd.DataFrame(np.zeros((10,3)),columns=["c1","c2","c3"])
  
fig, axs = plt.subplots()
clust_data = df.values
collabel = df.columns
axs.axis('tight')
axs.axis('off')
the_table = axs.table(cellText=clust_data,colLabels=collabel,loc='center')
plt.show()

