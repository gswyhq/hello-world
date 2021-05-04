import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Base64;
import com.alibaba.fastjson.*;

public class HttpURLConnectionHelper {

    public static String sendRequest(String urlParam,String requestType) {

        HttpURLConnection con = null;  

        BufferedReader buffer = null; 
        StringBuffer resultBuffer = null;  

        try {
            URL url = new URL(urlParam); 
            
            String name = "admin";

            String password = "admin";

            String authString = name + ":" + password;

            System.out.println("auth string: " + authString);

            Base64.Encoder encoder = Base64.getEncoder();
            byte[] textByte = authString.getBytes("UTF-8");
            //编码
            String authStringEnc = encoder.encodeToString(textByte);
            System.out.println(authStringEnc);
            
            //得到连接对象
            con = (HttpURLConnection) url.openConnection(); 
            //设置请求类型
            con.setRequestMethod(requestType);  
            //设置请求需要返回的数据类型和字符集类型
            con.setRequestProperty("Content-Type", "application/json;charset=GBK");  
            
            // 设置登录密码
            con.setRequestProperty("Authorization", "Basic "+ authStringEnc);
            
            //允许写出
            con.setDoOutput(true);
            //允许读入
            con.setDoInput(true);
            //不使用缓存
            con.setUseCaches(false);
            //得到响应码
            int responseCode = con.getResponseCode();

            if(responseCode == HttpURLConnection.HTTP_OK){
                //得到响应流
                InputStream inputStream = con.getInputStream();
                //将响应流转换成字符串
                resultBuffer = new StringBuffer();
                String line;
                buffer = new BufferedReader(new InputStreamReader(inputStream, "GBK"));
                while ((line = buffer.readLine()) != null) {
                    resultBuffer.append(line);
                }
                return resultBuffer.toString();
            }

        }
        catch(Exception e) {
            e.printStackTrace();
        }
        return "";
    }
    
    public static void main(String[] args) {

        String url ="http://12.15.62.130:8082/api/atlas/v2/search/attribute?attrName=qualifiedName&attrValuePrefix=sx_safe.pol_info_h.pol_sts&limit=10&offset=0&typeName=hive_column";
        String retStr = sendRequest(url,"GET");
        System.out.println("返回结果：\n"+retStr);
        
        JSONObject jsonObject = JSONObject.parseObject(retStr);
        List entities = jsonObject.getJSONArray("entities");
        int approximateCount = jsonObject.getIntValue("approximateCount");
        
        System.out.println(String.format("总记录数：%s", approximateCount));
        System.out.println(String.format("血缘关系：%s", entities));
        
        // 将返回结果写入文件；
        String filePath = "D:\\Users\\edwihe\\Downloads\\tag_shr_v_1w\\content.json";
        FileWriter fwriter = null;
     
        try {
            // true表示不覆盖原来的内容，而是加到文件的后面。若要覆盖原来的内容，直接省略这个参数就好
            fwriter = new FileWriter(filePath, false);
            fwriter.write(retStr);
        } catch (IOException ex) {
            ex.printStackTrace();
        } finally {
            try {
                fwriter.flush();
                fwriter.close();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
        
        System.out.println("完成！");
    }
}

