
第一步：编译java文件，并打包成jar
CMD窗口执行 run.bat文件完成将java程序打包为jar文件；run.bat文件内容如下：
:: 切换到项目目录：
cd D:\Users\user123

:: 将java文件编译成.class文件：
javac  -encoding UTF-8 -cp .m2/repository/org/apache/hadoop/hadoop-common/2.6.0/hadoop-common-2.6.0.jar;.m2/repository/org/apache/hive/hive-exec/2.3.7/hive-exec-2.3.7.jar   GPSConverter\GPS.java GPSConverter\GPSConverterUtils.java GPSConverter\*.java

:: 将.class文件打包成jar文件：
jar cvfm GPSConverter/Converter.jar GPSConverter/manf GPSConverter/*.class

:: 切换到cmd窗口执行
::start GPSConverter\run.bat

第二步：hive中添加jar包：
~/GPSConverter$ docker cp Converter.jar hadoop-hive2:/root/
hive> add jar /root/Converter.jar;
Added [/root/Converter.jar] to class path
Added resources: [/root/Converter.jar]

第三步：使用自定义的函数：
hive> create temporary function Bd09ToGcj02 as 'GPSConverter.Bd09ToGcj02';
hive> create temporary function Bd09ToWgs84 as 'GPSConverter.Bd09ToWgs84';
hive> create temporary function Gcj02ToBd09 as 'GPSConverter.Gcj02ToBd09';
hive> create temporary function Gcj02ToWgs84 as 'GPSConverter.Gcj02ToWgs84';
hive> create temporary function Wgs84ToBd09 as 'GPSConverter.Wgs84ToBd09';
hive> create temporary function Wgs84ToGcj02 as 'GPSConverter.Wgs84ToGcj02';
hive> select Wgs84ToBd09('113.963626,22.544535');
113.975025      22.547535
hive> select Wgs84ToGcj02('113.963626,22.544535');
113.968536      22.541564
hive> select Bd09ToWgs84('113.975025,22.547535');
113.963616      22.544523
hive> select Bd09ToWgs84('113.963626, 22.544535');
113.952204      22.541739
hive> select Gcj02ToWgs84('113.963626, 22.544535');
113.958716      22.547506
hive> select Wgs84ToGcj02('113.963626, 22.544535');
113.968536      22.541564
hive> select Bd09ToGcj02('113.963626, 22.544535');
113.957101      22.538752
hive> select Gcj02ToBd09('113.963626, 22.544535');
113.970134      22.550420
hive> select Wgs84ToBd09('113.963626, 22.544535');
113.975025      22.547535


hive> select province, city, area, name, lng, lat, adTable.wgs84_lng as wgs84_lng, adTable.wgs84_lat as wgs84_lat, address 
FROM hive_db1.tmp_house_addr_poi_shenzhen LATERAL VIEW Bd09ToWgs84(concat(lng,',',lat))  adTable AS `wgs84_lng`, `wgs84_lat` limit 5;
OK
广东    深圳    南山    沙河高尔夫别墅  113.97636682    22.53722209     113.964964      22.534184       广东省深圳市南山区沙河东路1号
广东    深圳    南山    名商高尔夫公寓  113.97712591    22.55816383     113.965711      22.555106       沙河东路111号
广东    深圳    福田    天安高尔夫分园  114.04599496    22.54303645     114.034396      22.539959       泰然一路
广东    深圳    龙岗    深圳信息职业技术学校学生公寓-C4栋       114.23120453    22.69882237     114.219879      22.695411       龙翔大道2188号(近体育新城)
广东    深圳    龙岗    深圳市第一职业技术学校学生宿舍-1号楼    114.52310716    22.605263600000004      114.511916      22.602284       鹏飞路49号

insert into table ggb_exchange_safe.addr_poi_2021 partition (y='2021',m='10',d='01')
select name,address, province, city, area, '0' as source, '商务住宅' as level1, '住宅小区' as level2, '' as level3, adTable.wgs84_lat as wgs84_lat, adTable.wgs84_lng as wgs84_lng
from hive_db1.tmp_house_addr_poi_shenzhen LATERAL VIEW Bd09ToWgs84(concat(lng,',',lat))  adTable AS `wgs84_lng`, `wgs84_lat` ;


省市区地址清洗：
第一步：
.py 编译为so
PCAClean/run_build.sh
PCAClean/run_java.sh

第二步：java调用：
cp PCAClean/PCAClean.java PCAClean.java
PCAClean.java 首行插入：package GPSConverter;
cp PCAClean/addtopost.pkl .

第三步：打包为jar;
start run.bat

第四步：创建临时函数
hive> add jar ***.jar
hive> create temporary function PClean as 'GPSConverter.PClean';

第五步：使用
hive> select PClean(';三水区;');

hive> select PClean(';三水区;');
OK
文件：/tmp/addtopost.pkl， 存在与否：True
读取文件 /tmp/addtopost.pkl 成功！
进入 JNI_API_test_function
这是一个python函数
python函数参数是： [;三水区;]
返回结果：['广东省', '佛山市', '三水区']
离开 JNI_API_test_function
广东省  佛山市  三水区

hive> select province, city, area, name, lng, lat,  address, t2.province1 as p1, t2.city1 as c1, t2.area1 as a1 FROM hive_db1.tmp_house_addr_poi_shenzhen LATERAL VIEW PClean(concat(province, ';', city, ';',
area)) t2 as `province1`, `city1`, `area1` limit 5;
广东    深圳    南山    沙河高尔夫别墅  113.97636682    22.53722209     广东省深圳市南山区沙河东路1号   广东省  深圳市  南山区
广东    深圳    南山    名商高尔夫公寓  113.97712591    22.55816383     沙河东路111号   广东省  深圳市  南山区
广东    深圳    福田    天安高尔夫分园  114.04599496    22.54303645     泰然一路        广东省  深圳市  福田区
广东    深圳    龙岗    深圳信息职业技术学校学生公寓-C4栋       114.23120453    22.69882237     龙翔大道2188号(近体育新城)      广东省  深圳市  龙岗区
广东    深圳    龙岗    深圳市第一职业技术学校学生宿舍-1号楼    114.52310716    22.605263600000004      鹏飞路49号      广东省  深圳市  龙岗区
hive> select province, city, area, name, lng, lat, adTable.wgs84_lng as wgs84_lng, adTable.wgs84_lat as wgs84_lat, address , t2.province1 as p1, t2.city1 as c1, t2.area1 as a1 FROM hive_db1.tmp_house_addr_p
oi_shenzhen LATERAL VIEW Bd09ToWgs84(concat(lng,',',lat))  adTable AS `wgs84_lng`, `wgs84_lat`  LATERAL VIEW PClean(concat(province, ';', city, ';', area)) t2 as `province1`, `city1`, `area1` limit 5;
广东    深圳    南山    沙河高尔夫别墅  113.97636682    22.53722209     113.964964      22.534184       广东省深圳市南山区沙河东路1号   广东省  深圳市  南山区
广东    深圳    南山    名商高尔夫公寓  113.97712591    22.55816383     113.965711      22.555106       沙河东路111号   广东省  深圳市  南山区
广东    深圳    福田    天安高尔夫分园  114.04599496    22.54303645     114.034396      22.539959       泰然一路        广东省  深圳市  福田区
广东    深圳    龙岗    深圳信息职业技术学校学生公寓-C4栋       114.23120453    22.69882237     114.219879      22.695411       龙翔大道2188号(近体育新城)      广东省  深圳市  龙岗区
广东    深圳    龙岗    深圳市第一职业技术学校学生宿舍-1号楼    114.52310716    22.605263600000004      114.511916      22.602284       鹏飞路49号      广东省  深圳市  龙岗区


