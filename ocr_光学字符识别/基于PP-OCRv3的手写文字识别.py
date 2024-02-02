
基于PP-OCRv3的手写文字识别
 
gswyhq@gswyhq-PC:~/github_project/PaddleOCR$ mv HW_Chinese/train_data .

gswyhq@gswyhq-PC:~/github_project/PaddleOCR$ less HW_Chinese/train.txt |shuf > HW_Chinese/train_shuf.txt 
gswyhq@gswyhq-PC:~/github_project/PaddleOCR$ wc -l HW_Chinese/train_shuf.txt 
239651 HW_Chinese/train_shuf.txt
gswyhq@gswyhq-PC:~/github_project/PaddleOCR$ head -n 210000 HW_Chinese/train_shuf.txt > ./train_data/train_list.txt
gswyhq@gswyhq-PC:~/github_project/PaddleOCR$ tail -n 29651 HW_Chinese/train_shuf.txt > ./train_data/val_list.txt

