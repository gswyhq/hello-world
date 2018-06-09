#! /bin/bash

vflag=off
filename=""
output=""

# 获取使用帮助：
#gswewf@gswewf-PC:~/yhb/es_search$ ./getopts_2.sh -h
#gswewf@gswewf-PC:~/yhb/es_search$ ./getopts_2.sh -help

function usage() {
        echo "使用方法:"
        echo "  getopts_2.sh [-h] [-v] [-f <filename>] [-o <filename>]"
        exit 1
}

# 在 while 循环中使用 getopts 解析命令行选项
# 要解析的选项有 -h、-v、-f 和 -o，其中 -f 和 -o 选项带有参数
# 字符串选项中第一个冒号表示 getopts 使用抑制错误报告模式

# optstring定义了四个有效选项字母: h, v, f, o
# 冒号（：）被放在了字母f和o后面，因为f选项、o选项各需要一个参数值

while getopts :hvf:o: opt
do
        case "$opt" in
                v)
                        vflag=on
                        ;;

                # 输入一个文件，并检测文件是否存在
                # gswewf@gswewf-PC:~/yhb/es_search$ ./getopts_2.sh -f asdfwef.pdf
                f)
                        filename=$OPTARG
                        if [ ! -f $filename ]
                        then
                                echo "源文件 $filename 不存在!"
                                exit
                        fi
                        ;;

                # 输出文件，并检测输出文件的路径是否存在
                # gswewf@gswewf-PC:~/yhb/es_search$ ./getopts_2.sh -o ./afwf/sdf
                o)
                        output=$OPTARG
                        # dirname命令可以取给定路径的目录部
                        if [ ! -d `dirname $output` ]
                        then
                                echo "选项提供的路径 `dirname $output` 不存在!"
                                exit
                        fi
                        ;;

                # 提供帮助选项
                # gswewf@gswewf-PC:~/yhb/es_search$ ./getopts_2.sh -h
                # gswewf@gswewf-PC:~/yhb/es_search$ ./getopts_2.sh -help
                h)
                        usage
                        exit
                        ;;
                :)
                        echo "选项 -$OPTARG 需要有个参数."
                        usage
                        exit 1
                        ;;
                ?)
                        echo "无效的选项: -$OPTARG"
                        usage
                        exit 2
                        ;;
        esac
done

echo "变量 valag 的值是: ${vflag}"

#getopts 是 shell 内建命令， getopt 是一个独立外部工具
#getopts 使用语法简单，getopt 使用语法复杂
#getopts 不支持长参数（长选项，如 --option）， getopt 支持
#getopts 不会重排所有参数的顺序，getopt会重排参数顺序 (getopts 的 shell 内置 OPTARG 这个变量，getopts 通过修改这个变量依次获取参数，而 getopt 必须使用 set 来重新设定位置参数，然后在 getopt 中使用 shift 来依次获取参数)
#如果某个参数中含有空格，那么这个参数就变成了多个参数。因此，基本上，如果参数中可能含有空格，那么必须用getopts(新版本的 getopt 也可以使用空格的参数，只是传参时，需要用 双引号 包起来)。