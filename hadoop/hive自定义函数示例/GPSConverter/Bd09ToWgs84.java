package GPSConverter;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
//import org.springframework.beans.factory.annotation.Autowired;
import java.util.ArrayList;

import GPSConverter.GPS;
import GPSConverter.GPSConverterUtils;

public class Bd09ToWgs84 extends GenericUDTF {

    public static GPSConverterUtils gpsConverterUtils;

    /** 初始化列名;
    初始化部分其实是给一个表头，也就是结果中的默认列，因为可以输出多行嘛，需要有列名，一个是列名，一个是列的类型：
     * add the column name
     * @param args
     * @return
     * @throws UDFArgumentException
     */
    @Override
    public StructObjectInspector initialize(ObjectInspector[] args) throws UDFArgumentException {
        if (args.length != 1) {
            throw new UDFArgumentLengthException("ExplodeMap takes only one argument");
        }
        if (args[0].getCategory() != ObjectInspector.Category.PRIMITIVE) {
            throw new UDFArgumentException("ExplodeMap takes string as a parameter");
        }

        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("wgs84_lng");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("wgs84_lat");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);


        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    /** 处理字符串部分
     * process the column
     * @param objects
     * @throws HiveException
     */
    public void process(Object[] objects) throws HiveException {

        String [] input = objects[0].toString().split(",");
        String bd09_lng = input[0];
        String bd09_lat = input[1];

        String[] result = new String[2];
//        result[0] = bd09_lng;
//        result[1] = bd09_lat;

        GPS wgs84 = gpsConverterUtils.bd09_To_Wgs84(Double.valueOf(bd09_lat), Double.valueOf(bd09_lng));
		//System.out.println(String.format("GPS84：%s, %s", wgs84.getLat(), wgs84.getLon()));
		result[0] =  String.format("%.6f", wgs84.getLon());
        result[1] = String.format("%.6f", wgs84.getLat());

        forward(result);  // 返回结果；
        return;
    }

    public void close() throws HiveException {

    }


}

