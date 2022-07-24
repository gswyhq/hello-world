
import time
import tqdm

for i in tqdm(range(1000)):
    time.sleep(0.0001)


100%|██████████| 1000/1000 [00:01<00:00, 521.68it/s]
Process finished with exit code 0

# tqdm, 默认是 0.1秒换一行，若不想要换行，即 输出一条进度条，这个时候可以设置 mininterval 参数；
# 比如设置成 5分钟换一行(但同时意味着5min不显示更新进度)：
for i in tqdm(range(1000), mininterval=5*60):
    time.sleep(0.1)

---------------------

with open('/home/gswyhq/Downloads/tqdm.log', 'w')as f:
    for i in tqdm(range(1000), file=f):
        time.sleep(0.01)
        
如果想对进度条进行更加详细的定制，可以实例化一个tqdm类的实例，然后使用它的方法来更好地发挥作用

实例化tqdm类时有一些其他的可能比较常用的参数：

iterable（第一个参数）：一个可迭代对象
desc：对进度条的描述，会显示在进度条前边
total：预期的总迭代次数（默认会等于iterable的总次数，如果可数的话）
ncols：总长度
mininterval：最小的更新时间间隔，默认为0.1
maxinterval：最大的更新时间间隔，默认为10
一个tqdm实例的常用方法：

set_description：设置显示在进度条前边的内容
set_postfix：设置显示在进度条后边的内容
update：对进度进行手动更新
close：关闭进度条实例，实际上，最好在使用完一个tqdm类的实例后使用 close 方法清理资源，就像使用open打开的文件一样，从而释放内存。

# 一个例子：

from tqdm import tqdm
import time, random

p_bar = tqdm(range(10), desc="A Processing Bar Sample: ", total=10, ncols=100)

for i in p_bar:
    time.sleep(random.random())

p_bar.close()


# 使用with语句
因为一个实例化的tqdm也需要在使用完毕后通过close方法清理资源，这和打开一个文件进行处理是很类似的，因此同样可以使用with语句，让其在执行完后自动清理，就不再需要使用close方法手动关闭了：

from tqdm import tqdm
import time, random

with tqdm(total=100) as p_bar:
    for i in range(50):
        time.sleep(random.random())
        p_bar.update(2)
        p_bar.set_description("Processing {}-th iteration".format(i+1))


