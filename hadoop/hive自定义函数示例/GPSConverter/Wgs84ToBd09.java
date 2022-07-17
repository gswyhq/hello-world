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

public class Wgs84ToBd09 extends GenericUDTF {

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
        fieldNames.add("bd09_lng");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("bd09_lat");
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
        String wgs84_lng = input[0];
        String wgs84_lat = input[1];

        String[] result = new String[2];
//        result[0] = wgs84_lng;
//        result[1] = wgs84_lat;

        GPS bd09 = gpsConverterUtils.wgs84_To_Bd09(Double.valueOf(wgs84_lat), Double.valueOf(wgs84_lng));
		//System.out.println(String.format("GPS84：%s, %s", bd09.getLat(), bd09.getLon()));
		result[0] =  String.format("%.6f", bd09.getLon());
        result[1] = String.format("%.6f", bd09.getLat());

        forward(result);  // 返回结果；
        return;
    }

    public void close() throws HiveException {

    }


}

