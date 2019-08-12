
import time
import tqdm

for i in tqdm(range(1000)):
    time.sleep(0.0001)


100%|██████████| 1000/1000 [00:01<00:00, 521.68it/s]
Process finished with exit code 0

---------------------

with open('/home/gswyhq/Downloads/tqdm.log', 'w')as f:
    for i in tqdm(range(1000), file=f):
        time.sleep(0.01)
        

