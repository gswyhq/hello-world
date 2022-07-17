//import PCAClean;

public class Demo {
    static PCAClean tester = new PCAClean();
    public static void main(String[] args) {
            //System.load("/root/test2/PCAClean.cpython-36m-x86_64-linux-gnu.so");
            //PCAClean tester = new PCAClean();
            //tester.loadLibrary();
	    tester.initModule();
	    
            String result = tester.pcaCleanFunction(";深圳;南山");
            System.out.println(String.format("java调用python函数的返回结果：%s", result));
	    String result2 = tester.pcaCleanFunction(";三水区;");
            tester.uninitModule();

            System.out.println(String.format("java调用python函数的返回结果：%s", result2));
								        }
}

