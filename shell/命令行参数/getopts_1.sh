#!/bin/bash

#gswyhq@gswyhq-PC:~/yhb/es_search$ ./getopts_1.sh -a
#gswyhq@gswyhq-PC:~/yhb/es_search$ ./getopts_1.sh -b

# 这里仅解析 -a 选项，选项字符串中的第一个字符为冒号(:)，表示抑制错误报告
while getopts ":a" opt
do
        case $opt in
                # 匹配 -a 选项
                a)
                        echo "The option -a was triggered!"
                        echo '${opt},这是传入的参数'
                        echo "${opt},这是传入的参数"
                        echo "也可以这样读${OPTARG}"
                        ;;
                # 匹配其他选项
                \?)
                        echo "无效的参数：${OPTARG}"
                        ;;
        esac
done

