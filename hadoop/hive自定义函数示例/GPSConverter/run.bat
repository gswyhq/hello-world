:: 切换到项目目录：

cd D:\Users\user123

:: 将java文件编译成.class文件：

javac  -encoding UTF-8 -cp .m2/repository/org/apache/hadoop/hadoop-common/2.6.0/hadoop-common-2.6.0.jar;.m2/repository/org/apache/hive/hive-exec/2.3.7/hive-exec-2.3.7.jar   GPSConverter\GPS.java GPSConverter\GPSConverterUtils.java GPSConverter\PCAClean.java GPSConverter\*.java

:: 将.class文件打包成jar文件：

jar cvfm GPSConverter/Converter.jar GPSConverter/manf GPSConverter/*.class GPSConverter/libPCACleanTest.so GPSConverter/addtopost.pkl

:: 切换到cmd窗口执行
::start GPSConverter\run.bat
