
public class Main {
    /**加载opencv链接库*/
    static {
        System.load("/usr/local/share/java/opencv4/libopencv_java400.so");
    }
    /**
     * @params path of bitmap
     * @params path of svg
     * */
    public static void main(String[]args) {

        /**
        * 调用 setParamter 指定参数
         * 调用 handle矢量化
         * 是否填充
        * */
        HandleProcess process=new HandleProcess();
        process.setParamter(50,false, HandleProcess.Type.NONE);
        process.handle("pic.jpg","dst.svg");
    }
}
