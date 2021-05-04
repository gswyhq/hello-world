

import java.io.File;
import java.io.FileOutputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class ReadWriteFile {



	public static String ReadFile() throws Exception{

        String path = "D:\\Users\\USERNAME\\Downloads\\tag_shr_v_1w\\v2_v6_party_id_col.csv";
        FileInputStream fileInputStream = new FileInputStream(path);

        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(fileInputStream));

        String line = null;
        List<String> strlist = new ArrayList<String>();
        
        while ((line = bufferedReader.readLine()) != null) {
//            System.out.println(line);
            strlist.add(line);
        }
        
        System.out.println(String.format("总共行数：%s", strlist.size()));
        
        String[] strarray = new String[strlist.size()];
        strlist.toArray(strarray );

        ArrayList inputA = new ArrayList(strlist.subList(0, strlist.size()/ 120));
        
        System.out.println(inputA);
        	    
        fileInputStream.close();
        return "ok";
    }

    public static String WriteFile() throws Exception{

        String path = "D:\\Users\\USERNAME\\Downloads\\tag_shr_v_1w\\hello.csv";

        File file = new File(path);

        String content = "hello,中国\n张三,广东\n";
        List<String> list2 = new ArrayList<String>(Arrays.asList("apple", "banana", "orange"));
        
        FileOutputStream fileOutputStream = new FileOutputStream(file);
        fileOutputStream.write(content.getBytes());
        for (String str1 : list2) {
        	 System.out.println(str1);
        	 content = str1 + "\n";
        	 fileOutputStream.write(content.getBytes());
        }
        fileOutputStream.close();
        return "ok";
    }

    public static void main(String[] args) {
    	try {
			ReadFile();
			WriteFile();
			System.out.println("读写完成");
		} catch (Exception e) {
			e.printStackTrace();
		}
    	
    }
}
