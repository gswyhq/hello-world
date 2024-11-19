


import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.List;

@RunWith(MockitoJUnitRunner.class)
public class ModelApiTest {

    @Test
    public void noticeRunTimeTest() throws Exception {

        String command = "netstat -n -p tcp | findstr \"7996\"";
//        command = "ipconfig /all|findstr IP";
        String osName = System.getProperty("os.name").toLowerCase();
        String[] cmdArr;
        if (osName.startsWith("win")) {
            cmdArr = new String[]{"cmd", "/c", command};
        } else if (osName.startsWith("linux")) {
            cmdArr = new String[]{"/bin/bash", "-c", command};
        } else {
            throw new Exception("不支持的操作系统："+osName);
        }
        List<String> commands = Arrays.asList(cmdArr);
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.command(commands);

        BufferedReader inputReader = null;
        BufferedReader errorReader = null;
        StringBuilder builder = new StringBuilder();
        try {
            Process process = processBuilder.start();
            //脚本执行输出信息
            inputReader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String str = null;
            while((str = inputReader.readLine()) != null){
                builder.append(str).append("\n");
            }

            //脚本执行异常时的输出信息
            errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error = null;
            while((error = inputReader.readLine()) != null){
                System.out.println(error);
            }
            try {
                if (0 != process.waitFor()) {
                    throw new Exception(String.format("execute [%s] fail!(%s)", String.join(" ", commands), ""));
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                if (inputReader != null) {
                    inputReader.close();
                }
                if (errorReader != null) {
                    errorReader.close();
                }
            }catch (Exception e){
                e.printStackTrace();
            }
        }

        System.out.println(builder.toString());

    }
}

