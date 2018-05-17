#!/bin/sh
# example: sh  delete_es_by_day.sh logstash-kettle-log logsdate 30
#有的时候我们在使用ES时，由于资源有限或业务需求，我们只想保存最近一段时间的数据，所以有如下脚本可以定时删除数据
#delete_es_by_day.sh
# 注解：脚本传入参数说明：1.索引名；2.日期字段名；3.保留最近几天数据，单位天；4.日期格式，可不输（默认形式20160101）

index_name=$1
daycolumn=$2
savedays=$3
format_day=$4

if [ ! -n "$savedays" ]; then
  echo "the args is not right,please input again...."
  exit 1
fi

if [ ! -n "$format_day" ]; then
   format_day='%Y%m%d'
fi

sevendayago=`date -d "-${savedays} day " +${format_day}`

#date +%Y%m%d         //显示现在天年月日
#date +%Y%m%d --date="+1 day"  //显示后一天的日期
#date +%Y%m%d --date="-1 day"  //显示前一天的日期
#date +%Y%m%d --date="-1 month"  //显示上一月的日期
#date +%Y%m%d --date="+1 month"  //显示下一月的日期
#date +%Y%m%d --date="-1 year"  //显示前一年的日期
#date +%Y%m%d --date="+1 year"  //显示下一年的日期
#root@727d1b1ba1ef:/intent# date +%Y%m%d%H%M%s
#2018051710441526525059
#root@727d1b1ba1ef:/intent# date +%Y%m%d%H%M%S
#20180517104424


curl -XDELETE "10.130.3.102:9200/${index_name}/_query?pretty" -d "
{
        "query": {
                "filtered": {
                        "filter": {
                                "bool": {
                                        "must": {
                                                "range": {
                                                        "${daycolumn}": {
                                                                "from": null,
                                                                "to": ${sevendayago},
                                                                "include_lower": true,
                                                                "include_upper": true
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
}"

echo "ok"