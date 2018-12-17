import java.util.ArrayList;

public class BitMap {
    public int w;
    public int h;
    public int size;
    public ArrayList arraybuffer;
    public byte[]data;

    public BitMap(int w,int h){
        this.w=w;
        this.h=h;
        this.size=w*h;
        this.arraybuffer=new ArrayList(size);
        this.data=new byte[size];

    }


   public boolean at(double x, double y) {
        return (x >= 0 && x < this.w && y >=0 && y < this.h) &&
                this.data[(int )(this.w * y + x)] == 1;
    };



    public Point index(int i) {
        Point point = new Point(0,0);
        point.y = (int)Math.floor(i / this.w);
        point.x = i - point.y * this.w;
        return point;
    };

    public void flip (double x, double y) {
        if (this.at(x, y)) {
            this.data[(int )(this.w * y + x)] = 0;
        } else {
            this.data[(int )(this.w * y + x)] = 1;
        }
    };

    public BitMap copy () {
        BitMap bm = new BitMap(this.w, this.h);
        for (int i = 0; i < this.size; i++) {
            bm.data[i] = this.data[i];
        }
        return bm;
    };


}
