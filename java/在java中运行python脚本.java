

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class RunPython {
    
    public static void main(String[] args) { 
        try {
           String a=getPara("car").substring(1),b="D34567",c="LJeff34",d="iqngfao";
           String url="http://blog.csdn.net/thorny_v/article/details/61417386";
           System.out.println("start;"+url);
           String[] args1 = new String[] { "python", "D:\\Users\\name\\Documents\\java\\test_java.py", url, a, b, c, d}; 
           Process pr=Runtime.getRuntime().exec(args1);
           BufferedReader in = new BufferedReader(new InputStreamReader(
             pr.getInputStream(), "GBK"));
           String line;
           while ((line = in.readLine()) != null) {
            System.out.println(line);
           }
           in.close();
           pr.waitFor();
           System.out.println("end");
          } 
        catch (Exception e) {
           e.printStackTrace();
          }
        System.out.println("完成；");
        }
    
    private static String getPara(String string) {
        return string+"123";
    }


}
