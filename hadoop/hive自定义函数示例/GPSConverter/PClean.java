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

import GPSConverter.PCAClean;

public class PClean extends GenericUDTF {

    static PCAClean pcaClean = new PCAClean();
    static {
        System.out.println(String.format("PClean 初始化"));
        pcaClean.initModule();
    }
//    public static void main(String[] args) {
//        //System.load("/root/test2/PCAClean.cpython-36m-x86_64-linux-gnu.so");
//        //PCAClean pcaClean = new PCAClean();
//        //pcaClean.loadLibrary();
//        pcaClean.initModule();
//
//        String result = pcaClean.pcaCleanFunction(";深圳;南山");
//        System.out.println(String.format("java调用python函数的返回结果：%s", result));
//        String result2 = pcaClean.pcaCleanFunction(";三水区;");
//        pcaClean.uninitModule();
//
//        System.out.println(String.format("java调用python函数的返回结果：%s", result2));
//    }

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
        fieldNames.add("province");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("city");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        fieldNames.add("area");
        fieldOIs.add(PrimitiveObjectInspectorFactory.javaStringObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    /** 处理字符串部分
     * process the column
     * @param objects
     * @throws HiveException
     */
    public void process(Object[] objects) throws HiveException {

//        String [] input = objects[0].toString().split(";");
//        String gcj02_lng = input[0];
//        String gcj02_lat = input[1];
//
        String[] result = new String[3];
//        result[0] = gcj02_lng;
//        result[1] = gcj02_lat;


        String resultStr = pcaClean.pcaCleanFunction(objects[0].toString());


		//System.out.println(String.format("返回结果 %s", result));
        String [] resultList = resultStr.split(";");
		result[0] = resultList[0];
        result[1] = resultList[1];
        result[2] = resultList[2];

        forward(result);  // 返回结果；
        return;
    }

    public void close() throws HiveException {
//        System.out.println("close");
//        pcaClean.uninitModule();
    }
}

