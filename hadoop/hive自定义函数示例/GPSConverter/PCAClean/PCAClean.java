import java.io.*;

public class PCAClean {
    public native void initModule();
    public native void uninitModule();
    public native String pcaCleanFunction(String param);
    // public synchronized static void loadLibrary() throws IOException {
    static {
    // 方法1.使用绝对路径加载，System.load("absolutePath/library.so");
    //    String SO_FILE_PATH = "/root/test2/Test.cpython-36m-x86_64-linux-gnu.so";
    //    System.load(SO_FILE_PATH);
    //    方法2.将链接库放在java.library.path指定的目录下，System.loadLibrary("library");
    //    ~$ mv "/root/test2/Test.cpython-36m-x86_64-linux-gnu.so" libPCACleanTest.so
    //    String SO_FILE_PATH = "Test";
    //	System.loadLibrary(SO_FILE_PATH);

    //    System.out.println(String.format("加载so文件完成: %s", SO_FILE_PATH));
       
    String libName = "libPCACleanTest";
    String BIN_LIB = ".";
    String addrName= "addtopost.pkl";
    String systemType = System.getProperty("os.name");
    String libExtension = (systemType.toLowerCase().indexOf("win")!=-1) ? ".dll" : ".so";
    String libFullName = libName + libExtension;
    String nativeTempDir = System.getProperty("java.io.tmpdir");
    InputStream in = null;
    BufferedInputStream reader = null;
    FileOutputStream writer = null;
    InputStream addrin = null;
    BufferedInputStream reader2 = null;
    FileOutputStream writer2 = null;
    File extractedLibFile = new File(nativeTempDir+File.separator+libFullName);
    File saveAddrFile = new File(nativeTempDir+File.separator+addrName);
    if(!extractedLibFile.exists()){
        try {
            in = PCAClean.class.getResourceAsStream(BIN_LIB +File.separator+ libFullName);
            addrin = PCAClean.class.getResourceAsStream(BIN_LIB +File.separator+ addrName);
            if(in==null) {
                in = PCAClean.class.getResourceAsStream(libFullName);
                addrin = PCAClean.class.getResourceAsStream(addrName);
            }
//            PCAClean.class.getResource(libFullName);
            reader = new BufferedInputStream(in);
            writer = new FileOutputStream(extractedLibFile);
            byte[] buffer = new byte[1024];
            while (reader.read(buffer) > 0){
                writer.write(buffer);
                buffer = new byte[1024];
            }
            reader2 = new BufferedInputStream(addrin);
            writer2 = new FileOutputStream(saveAddrFile);
            byte[] buffer2 = new byte[1024];
            while (reader2.read(buffer2) > 0){
                writer2.write(buffer2);
                buffer2 = new byte[1024];
            }

        } catch (IOException e){
            e.printStackTrace();
        } finally {
            if(in!=null)
                try{
			in.close();
		}catch(IOException e1){
                    System.out.println("本地库加载失败1");
                }
            if(writer!=null)
                try{writer.close();
		}catch(IOException e2){
			System.out.println("本地库加载失败2");
                }
        }
    }
    System.load(extractedLibFile.toString());
    System.out.println(String.format("加载成功：%s", extractedLibFile.toString()));
    }
}


