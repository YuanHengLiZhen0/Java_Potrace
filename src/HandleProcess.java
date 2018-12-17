import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.script.Invocable;
import javax.script.ScriptEngine;
import javax.script.ScriptEngineManager;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class HandleProcess implements  Potrace{

   private BitMap bm;
   private BitMap bm1;
   private Info info=new Info();
   private List<Path> pathlist = new ArrayList<>();
   private boolean inShape;
   private Type opt_type;
   enum Type{
       WIHTE,NONE;
   }
/**
 * @param minSize :最小轮廓面积,低于该面积的舍弃
 * @param inShape:是否需要包含内轮廓
 * */
    public   void setParamter(int minSize,boolean inShape,Type type){
        info.turdsize=minSize;
        this.inShape=inShape;
        this.opt_type=type;
    }

    public  void  handle(String in,String out){


            Mat src= Imgcodecs.imread(in,0);
            int width=src.cols();
            int height=src.rows();
            byte []bytes=new byte[width*height];
            src.get(0,0,bytes);
            /**生成01数组*/
            for(int i=0;i<width;i++){
                for(int j=0;j<height;j++){
                    bytes[j*width+i]=(byte)(bytes[j*width+i]>=0?1:0);
                }
            }
           String obj= process1(bytes,width,height);
            System.out.println(obj);
          try{
        File outFile=new File(out);
        FileOutputStream outputStream=new FileOutputStream(outFile);
        byte[]buffer=obj.getBytes();
        int len=buffer.length;
        int d=len/(1024*80);
        int k=0;
        while(k<=d) {
            if(k==d)outputStream.write(buffer,k++,len%(1024*80));
            else outputStream.write(buffer,k++,1024*80);
            outputStream.flush();
          }
        outputStream.close();

        } catch (Exception e) {
        e.printStackTrace();
        }

    }public  void  process(String in,String out){

            Mat src= Imgcodecs.imread(in,0);
            int width=src.cols();
            int height=src.rows();
            byte []bytes=new byte[width*height];
            src.get(0,0,bytes);
            /**生成01数组*/
            for(int i=0;i<height;i++){
                for(int j=0;j<width;j++){
                    bytes[i*width+j]=(byte)(bytes[i*width+j]>=0?1:0);
                }
            }
            ScriptEngineManager manager = new ScriptEngineManager();
            ScriptEngine engine = manager.getEngineByName("js");
            try {
                engine.eval(new FileReader("test.js"));
                if (engine instanceof Invocable) {
                    Invocable invocable = (Invocable) engine;
                    Object obj = invocable.invokeFunction("process", bytes, width, height);
                    System.out.println(obj);
                    File outFile = new File(out);
                    FileOutputStream outputStream = new FileOutputStream(outFile);
                    byte[] buffer = obj.toString().getBytes();
                    int len = buffer.length;
                    int d = len / (1024 * 80);
                    int k = 0;
                    while (k <= d) {
                        if (k == d) outputStream.write(buffer, k++, len % (1024 * 80));
                        else outputStream.write(buffer, k++, 1024 * 80);
                        outputStream.flush();
                    }
                    outputStream.close();

                }
            }catch (Exception e) {
        e.printStackTrace();
        }


    }

    @Override
    public String process1(byte[] bytes, int width, int height) {
        loadBm(bytes,width,height);
        bmToPathlist();
        processPath();
        return getSVG(1,opt_type);
    }

    private String bezier(int i,int size,Curve curve) {
        String b = "C " + (curve.c[i * 3 + 0].x * size) + " " +
                (curve.c[i * 3 + 0].y * size)+ ",";
        b += (curve.c[i * 3 + 1].x * size) + " " +
                (curve.c[i * 3 + 1].y * size) + ",";
        b += (curve.c[i * 3 + 2].x * size) + " " +
                (curve.c[i * 3 + 2].y * size)+ " ";
        return b;
    }

    private String segment(int i,int size,Curve curve) {
        String s = "L " + (curve.c[i * 3 + 1].x * size) + " " +
                (curve.c[i * 3 + 1].y * size) + " ";
        s += (curve.c[i * 3 + 2].x * size)+ " " +
                (curve.c[i * 3 + 2].y * size) + " ";
        return s;
    }
    private  String path(int i ,int size ,Curve curve) {

        int n = curve.n;
        String p = "M" + (curve.c[(n - 1) * 3 + 2].x * size) +
                " " + (curve.c[(n - 1) * 3 + 2].y * size) + " ";
        for (i = 0; i < n; i++) {
            if (curve.tag[i].equals("CURVE")) {
                p += bezier(i,size,curve);
            } else if (curve.tag[i].equals("CORNER")) {
                p += segment(i,size,curve);
            }
        }
        //p +=
        return p;
    }
    private String getSVG(int size, Type opt_type) {


        int  w = bm.w * size, h = bm.h * size,
                len = pathlist.size();
                Curve c;
                int i;
        String strokec, fillc, fillrule;

        String svg = "<svg id=\"svg\" version=\"1.1\" width=\"" + w + "\" height=\""+ h + "\" xmlns=\"http://www.w3.org/2000/svg\">";
        svg += "<path d=\"";
        for (i = 0; i < len; i++) {
            if(inShape==false&&pathlist.get(i).sign=='+')continue;
            c = pathlist.get(i).curve;
            svg += path(i,size,c);
        }
        if (opt_type ==Type.WIHTE) {
            strokec = "black";
            fillc = "white";
            fillrule = "";
        } else {
            strokec = "black";
            fillc = "none";
            fillrule = " fill-rule=\"nonzero\"";
        }
        svg += "\" stroke=\"" + strokec + "\" fill=\"" + fillc + "\"" + fillrule + "/></svg>";

        return svg;
    }


    private int  mod(double a, double n) {
        return (int) ((a >= n) ? (a % n) : (a>=0 ? a : (n-1-(-1-a) % n)));
    }


    private double  xprod(Point p1, Point p2) {
        return p1.x * p2.y - p1.y * p2.x;
    }

    private boolean cyclic(double a, double b, double c) {
        if (a <= c) {
            return (a <= b && b < c);
        } else {
            return (a <= b || b < c);
        }
    }

    private double sign(double i) {
        return i > 0 ? 1 : i < 0 ? -1 : 0;
    }
    private double quadform(Quad Q, Point w) {
        double[] v = new double[3];
        int i, j;
        double sum;

        v[0] = w.x;
        v[1] = w.y;
        v[2] = 1;
        sum = 0.0;

        for (i=0; i<3; i++) {
            for (j=0; j<3; j++) {
                sum += v[i] * (Q.at(i, j)) * v[j];
            }
        }
        return sum;
    }

    private Point interval(double lambda, Point a, Point b) {
        Point res = new Point(0,0);

        res.x =  (a.x + lambda * (b.x - a.x));
        res.y =  (a.y + lambda * (b.y - a.y));
        return res;
    }
    private   Point dorth_infty(Point p0,Point  p2) {
        Point r = new Point(0,0);
        r.y = sign(p2.x - p0.x);
        r.x = -sign(p2.y - p0.y);
        return r;
    }

    private double ddenom(Point p0, Point p2) {
        Point r = dorth_infty(p0, p2);

        return r.y * (p2.x - p0.x) - r.x * (p2.y - p0.y);
    }

    private double dpara( Point p0, Point p1, Point p2) {
        double  x1, y1, x2, y2;
        x1 = p1.x - p0.x;
        y1 = p1.y - p0.y;
        x2 = p2.x - p0.x;
        y2 = p2.y - p0.y;
        return x1 * y2 - x2 * y1;
    }


    private double cprod(Point p0, Point p1, Point p2,Point  p3) {
        double  x1, y1, x2, y2;
        x1 = p1.x - p0.x;
        y1 = p1.y - p0.y;
        x2 = p3.x - p2.x;
        y2 = p3.y - p2.y;
        return x1 * y2 - x2 * y1;
    }
    private double iprod(Point p0, Point p1, Point p2) {
        double  x1, y1, x2, y2;

        x1 = p1.x - p0.x;
        y1 = p1.y - p0.y;
        x2 = p2.x - p0.x;
        y2 = p2.y - p0.y;

        return x1*x2 + y1*y2;
    }

    private double iprod1( Point p0,Point  p1,Point p2,Point p3) {
        double  x1, y1, x2, y2;

        x1 = p1.x - p0.x;
        y1 = p1.y - p0.y;
        x2 = p3.x - p2.x;
        y2 = p3.y - p2.y;

        return x1 * x2 + y1 * y2;
    }
    private double ddist(Point p, Point q) {
        return Math.sqrt((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y));
    }

    private Point bezier(double t,Point p0,Point p1,Point p2,Point p3) {
        double s = 1 - t; Point res = new Point(0,0);
        res.x = s*s*s*p0.x + 3*(s*s*t)*p1.x + 3*(t*t*s)*p2.x + t*t*t*p3.x;
        res.y = s*s*s*p0.y + 3*(s*s*t)*p1.y + 3*(t*t*s)*p2.y + t*t*t*p3.y;
        return res;
    }

    private  double tangent(Point p0,Point p1,Point p2,Point p3,Point q0,Point q1) {
        double A, B, C, a, b, c, d;
        double s, r1, r2;

        A = cprod(p0, p1, q0, q1);
        B = cprod(p1, p2, q0, q1);
        C = cprod(p2, p3, q0, q1);

        a = A - 2 * B + C;
        b = -2 * A + 2 * B;
        c = A;

        d = b * b - 4 * a * c;

        if (a==0 || d<0) {
            return -1.0;
        }

        s =  Math.sqrt(d);

        r1 = (-b + s) / (2 * a);
        r2 = (-b - s) / (2 * a);

        if (r1 >= 0 && r1 <= 1) {
            return r1;
        } else if (r2 >= 0 && r2 <= 1) {
            return r2;
        } else {
            return -1.0;
        }
    }
    private  void calcSums(Path path) {
        int  i; double x, y;
        path.x0 = path.pt.get(0).x;
        path.y0 = path.pt.get(0).y;

        path.sums = new ArrayList<Sum>();
        List <Sum>s = path.sums;
        s.add(new Sum(0, 0, 0, 0, 0));
        for(i = 0; i < path.len; i++){
            x = path.pt.get(i).x - path.x0;
            y = path.pt.get(i).y - path.y0;
            s.add(new Sum(s.get(i).x + x, s.get(i).y + y, s.get(i).xy + x * y,
                    s.get(i).x2 + x * x, s.get(i).y2 + y * y));
        }
    }
    private  void calcLon(Path path) {
        int n = path.len;
        List<Point>pt = path.pt;
        int dir;
        int[] pivk = new int[n], nc = new int[n], ct = new int[4];
        path.lon = new  double[n];

        Point [] constraint = {new Point(0,0), new Point(0,0)};
        Point cur = new Point(0,0),
                off = new Point(0,0),
                dk = new Point(0,0);
                int foundk;
        int  i,  k1;
        double a, b, c, d,j;int k = 0;
        for(i = n - 1; i >= 0; i--){
            if (path.pt.get(i).x != path.pt.get(k).x && path.pt.get(i).y != path.pt.get(k).y) {
                k = i + 1;
            }
            nc[i] = k;
        }

        for (i = n - 1; i >= 0; i--) {
            ct[0] = ct[1] = ct[2] = ct[3] = 0;
            dir = (int) ((3 + 3 * (path.pt.get(mod(i + 1, n)).x - path.pt.get(i).x) +
                                (path.pt.get(mod(i + 1, n)).y - path.pt.get(i).y)) / 2);
            ct[dir]++;

            constraint[0].x = 0;
            constraint[0].y = 0;
            constraint[1].x = 0;
            constraint[1].y = 0;

            k = nc[i];
            k1 = i;
            while (true) {
                foundk = 0;
                dir = (int) ((3 + 3 * sign(path.pt.get(k).x - path.pt.get(k1).x) +
                                        sign(path.pt.get(k).y - path.pt.get(k1).y)) / 2);
                ct[dir]++;

                if (ct[0] ==1&& ct[1]==1 && ct[2]==1 && ct[3]==1) {
                    pivk[i] = k1;
                    foundk = 1;
                    break;
                }

                cur.x = path.pt.get(k).x - path.pt.get(i).x;
                cur.y = path.pt.get(k).y - path.pt.get(i).y;

                if (xprod(constraint[0], cur) < 0 || xprod(constraint[1], cur) > 0) {
                    break;
                }

                if (Math.abs(cur.x) <= 1 && Math.abs(cur.y) <= 1) {

                } else {
                    off.x = cur.x + ((cur.y >= 0 && (cur.y > 0 || cur.x < 0)) ? 1 : -1);
                    off.y = cur.y + ((cur.x <= 0 && (cur.x < 0 || cur.y < 0)) ? 1 : -1);
                    if (xprod(constraint[0], off) >= 0) {
                        constraint[0].x = off.x;
                        constraint[0].y = off.y;
                    }
                    off.x = cur.x + ((cur.y <= 0 && (cur.y < 0 || cur.x < 0)) ? 1 : -1);
                    off.y = cur.y + ((cur.x >= 0 && (cur.x > 0 || cur.y < 0)) ? 1 : -1);
                    if (xprod(constraint[1], off) <= 0) {
                        constraint[1].x = off.x;
                        constraint[1].y = off.y;
                    }
                }
                k1 = k;
                k = nc[k1];
                if (!cyclic(k, i, k1)) {
                    break;
                }
            }
            if (foundk == 0) {
                dk.x = sign(path.pt.get(k).x-path.pt.get(k1).x);
                dk.y = sign(path.pt.get(k).y-path.pt.get(k1).y);
                cur.x = path.pt.get(k1).x - path.pt.get(i).x;
                cur.y = path.pt.get(k1).y - path.pt.get(i).y;

                a = xprod(constraint[0],cur);
                b = xprod(constraint[0],dk);
                c = xprod(constraint[1],cur);
                d = xprod(constraint[1],dk);

                j = 10000000;
                if (b < 0) {
                    j =  Math.floor(a / -b);
                }
                if (d > 0) {
                    j = Math.min(j, Math.floor(-c / d));
                }
                pivk[i] = mod(k1+j,n);
            }
        }

        j=pivk[n-1];
        path.lon[n-1]=j;
        for (i=n-2; i>=0; i--) {
            if (cyclic(i+1,pivk[i],j)) {
                j=pivk[i];
            }
            path.lon[i]=j;
        }

        for (i=n-1; cyclic(mod(i+1,n),j,path.lon[i]); i--) {
            path.lon[i] = j;
        }
    }

    private  double penalty3(Path path, int i, int j) {

        int  n = path.len;
        List<Point>pt = path.pt;
          List <Sum>sums = path.sums;
        double x, y, xy, x2, y2,
                k;
        double a, b, c, s;
        double     px, py, ex, ey;
             int   r = 0;
        if (j>=n) {
            j -= n;
            r = 1;
        }

        if (r == 0) {
            x = sums.get(j+1).x - sums.get(i).x;
            y = sums.get(j+1).y - sums.get(i).y;
            x2 = sums.get(j+1).x2 - sums.get(i).x2;
            xy = sums.get(j+1).xy - sums.get(i).xy;
            y2 = sums.get(j+1).y2 - sums.get(i).y2;
            k = j+1 - i;
        } else {
            x = sums.get(j+1).x - sums.get(i).x + sums.get(n).x;
            y = sums.get(j+1).y - sums.get(i).y + sums.get(n).y;
            x2 = sums.get(j+1).x2 - sums.get(i).x2 + sums.get(n).x2;
            xy = sums.get(j+1).xy - sums.get(i).xy + sums.get(n).xy;
            y2 = sums.get(j+1).y2 - sums.get(i).y2 + sums.get(n).y2;
            k = j+1 - i + n;
        }

        px =  ((pt.get(i).x + pt.get(j).x) / 2.0 - pt.get(0).x);
        py = ((pt.get(i).y + pt.get(j).y) / 2.0 - pt.get(0).y);
        ey = (pt.get(j).x - pt.get(i).x);
        ex = -(pt.get(j).y - pt.get(i).y);

        a = ((x2 - 2*x*px) / k + px*px);
        b = ((xy - x*py - y*px) / k + px*py);
        c = ((y2 - 2*y*py) / k + py*py);

        s = ex*ex*a + 2*ex*ey*b + ey*ey*c;

        return Math.sqrt(s);
    }
    private void  bestPolygon(Path path) {
        int  i, j, m, k, n = path.len;
               double [] pen = new double [n+1];
        int []  prev = new int [n+1];
        int [] clip0 = new int [n];
        int []   clip1 = new int [n+1];
        int []    seg0 = new int [n+1];
        int []     seg1 = new int [n+1];
             double    thispen, best;int c;

        for (i=0; i<n; i++) {
            c = mod(path.lon[mod(i-1,n)]-1,n);
            if (c == i) {
                c = mod(i+1,n);
            }
            if (c < i) {
                clip0[i] = n;
            } else {
                clip0[i] = c;
            }
        }

        j = 1;
        for (i=0; i<n; i++) {
            while (j <= clip0[i]) {
                clip1[j] = i;
                j++;
            }
        }

        i = 0;
        for (j=0; i<n; j++) {
            seg0[j] = i;
            i = clip0[i];
        }
        seg0[j] = n;
        m = j;

        i = n;
        for (j=m; j>0; j--) {
            seg1[j] = i;
            i = clip1[i];
        }
        seg1[0] = 0;

        pen[0]=0;
        for (j=1; j<=m; j++) {
            for (i=seg1[j]; i<=seg0[j]; i++) {
                best = -1;
                for (k=seg0[j-1]; k>=clip1[i]; k--) {
                    thispen = penalty3(path, k, i) + pen[k];
                    if (best < 0 || thispen < best) {
                        prev[i] = k;
                        best = thispen;
                    }
                }
                pen[i] = best;
            }
        }
        path.m = m;
        path.po = new int [m];

        for (i=n, j=m-1; i>0; j--) {
            i = prev[i];
            path.po[j] = i;
        }
    }


    private void  pointslope(Path path, int i,int  j,Point ctr,Point dir) {

        int  n = path.len;List<Sum> sums = path.sums;
               double  x, y, x2, xy, y2;int k;
               double a, b, c, lambda2  ,l;int r=0;

        while (j>=n) {
            j-=n;
            r+=1;
        }
        while (i>=n) {
            i-=n;
            r-=1;
        }
        while (j<0) {
            j+=n;
            r-=1;
        }
        while (i<0) {
            i+=n;
            r+=1;
        }

        x = sums.get(j+1).x-sums.get(i).x+r*sums.get(n).x;
        y = sums.get(j+1).y-sums.get(i).y+r*sums.get(n).y;
        x2 = sums.get(j+1).x2-sums.get(i).x2+r*sums.get(n).x2;
        xy = sums.get(j+1).xy-sums.get(i).xy+r*sums.get(n).xy;
        y2 = sums.get(j+1).y2-sums.get(i).y2+r*sums.get(n).y2;
        k = j+1-i+r*n;

        ctr.x = x/k;
        ctr.y = y/k;

        a = (x2-x*x/k)/k;
        b = (xy-x*y/k)/k;
        c = (y2-y*y/k)/k;

        lambda2 =  ((a+c+Math.sqrt((a-c)*(a-c)+4*b*b))/2);

        a -= lambda2;
        c -= lambda2;

        if (Math.abs(a) >= Math.abs(c)) {
            l = Math.sqrt(a*a+b*b);
            if (l!=0) {
                dir.x =  (-b/l);
                dir.y =  (a/l);
            }
        } else {
            l =  Math.sqrt(c*c+b*b);
            if (l!=0) {
                dir.x =  (-c/l);
                dir.y = (b/l);
            }
        }
        if (l==0) {
            dir.x = dir.y = 0;
        }
    }
    private void adjustVertices(Path path) {
        int  m = path.m; int []po = path.po;int n = path.len;
        List<Point>pt = path.pt;
             double    x0 = path.x0;double y0 = path.y0;
             Point []   ctr = new Point [m], dir = new Point[m];
        Quad []    q = new Quad [m];
        double []    v = new double [3];double d;int i, j, k, l;
        Point  s = new Point(0,0);

        path.curve = new Curve(m);

        for (i=0; i<m; i++) {
            j = po[mod(i+1,m)];
            j = mod(j-po[i],n)+po[i];
            ctr[i] = new Point(0,0);
            dir[i] = new Point(0,0);
            pointslope(path, po[i], j, ctr[i], dir[i]);
        }

        for (i=0; i<m; i++) {
            q[i] = new Quad();
            d = dir[i].x * dir[i].x + dir[i].y * dir[i].y;
            if (d == 0.0) {
                for (j=0; j<3; j++) {
                    for (k=0; k<3; k++) {
                        q[i].data[j * 3 + k] = 0;
                    }
                }
            } else {
                v[0] = dir[i].y;
                v[1] = -dir[i].x;
                v[2] = - v[1] * ctr[i].y - v[0] * ctr[i].x;
                for (l=0; l<3; l++) {
                    for (k=0; k<3; k++) {
                        q[i].data[l * 3 + k] = v[l] * v[k] / d;
                    }
                }
            }
        }

        Quad Q; Point w;
        double dx, dy, det, min, cand, xmin, ymin, z;
        for (i=0; i<m; i++) {
            Q = new Quad();
            w = new Point(0,0 );

            s.x = pt.get(po[i]).x-x0;
            s.y = pt.get(po[i]).y-y0;

            j = mod(i-1,m);

            for (l=0; l<3; l++) {
                for (k=0; k<3; k++) {
                    Q.data[l * 3 + k] = q[j].at(l, k) + q[i].at(l, k);
                }
            }

            while(true) {

                det = Q.at(0, 0)*Q.at(1, 1) - Q.at(0, 1)*Q.at(1, 0);
                if (det != 0.0) {
                    w.x = (-Q.at(0, 2)*Q.at(1, 1) + Q.at(1, 2)*Q.at(0, 1)) / det;
                    w.y = ( Q.at(0, 2)*Q.at(1, 0) - Q.at(1, 2)*Q.at(0, 0)) / det;
                    break;
                }

                if (Q.at(0, 0)>Q.at(1, 1)) {
                    v[0] = -Q.at(0, 1);
                    v[1] = Q.at(0, 0);
                } else if (Q.at(1, 1)>0) {
                    v[0] = -Q.at(1, 1);
                    v[1] = Q.at(1, 0);
                } else {
                    v[0] = 1;
                    v[1] = 0;
                }
                d = v[0] * v[0] + v[1] * v[1];
                v[2] = - v[1] * s.y - v[0] * s.x;
                for (l=0; l<3; l++) {
                    for (k=0; k<3; k++) {
                        Q.data[l * 3 + k] += v[l] * v[k] / d;
                    }
                }
            }
            dx = Math.abs(w.x-s.x);
            dy = Math.abs(w.y-s.y);
            if (dx <= 0.5 && dy <= 0.5) {
                path.curve.vertex[i] = new Point(w.x+x0, w.y+y0);
                continue;
            }

            min = quadform(Q, s);
            xmin = s.x;
            ymin = s.y;

            if (Q.at(0, 0) != 0.0) {
                for (z=0; z<2; z++) {
                    w.y = (s.y-0.5+z);
                    w.x = - (Q.at(0, 1) * w.y + Q.at(0, 2)) / Q.at(0, 0);
                    dx = Math.abs(w.x-s.x);
                    cand =  quadform(Q, w);
                    if (dx <= 0.5 && cand < min) {
                        min = cand;
                        xmin = w.x;
                        ymin = w.y;
                    }
                }
            }

            if (Q.at(1, 1) != 0.0) {
                for (z=0; z<2; z++) {
                    w.x =  (s.x-0.5+z);
                    w.y = - (Q.at(1, 0) * w.x + Q.at(1, 2)) / Q.at(1, 1);
                    dy = Math.abs(w.y-s.y);
                    cand =  quadform(Q, w);
                    if (dy <= 0.5 && cand < min) {
                        min = cand;
                        xmin = w.x;
                        ymin = w.y;
                    }
                }
            }

            for (l=0; l<2; l++) {
                for (k=0; k<2; k++) {
                    w.x =  (s.x-0.5+l);
                    w.y =  (s.y-0.5+k);
                    cand =  quadform(Q, w);
                    if (cand < min) {
                        min = cand;
                        xmin = w.x;
                        ymin = w.y;
                    }
                }
            }

            path.curve.vertex[i] = new Point(xmin + x0, ymin + y0);
        }
    }

    private void  reverse(Path path) {
        Curve curve = path.curve;
        int m = curve.n;
        Point []v = curve.vertex;
        int i, j;
                Point tmp=new Point(0,0);

        for (i=0, j=m-1; i<j; i++, j--) {
            tmp = v[i];
            v[i] = v[j];
            v[j] = tmp;
        }
    }

    private void  smooth( Path path) {
        int  m = path.curve.n;
        Curve curve = path.curve;

        int  i, j, k; double denom;
                double dd,alpha;
              Point p2, p3, p4;

        for (i=0; i<m; i++) {
            j = mod(i+1, m);
            k = mod(i+2, m);
            p4 = interval(1/2.0, curve.vertex[k], curve.vertex[j]);

            denom = ddenom(curve.vertex[i], curve.vertex[k]);
            if (denom != 0) {
                dd = dpara(curve.vertex[i], curve.vertex[j], curve.vertex[k]) / denom;
                dd = Math.abs(dd);
                alpha = dd>1 ?  (1 - 1.0 / dd) : 0;
                alpha =  (alpha / 0.75);
            } else {
                alpha =  (4/3.0);
            }
            curve.alpha0[j] = alpha;

            if (alpha >= info.alphamax) {
                curve.tag[j] = "CORNER";
                curve.c[3 * j + 1] = curve.vertex[j];
                curve.c[3 * j + 2] = p4;
            } else {
                if (alpha < 0.55) {
                    alpha = 0.55;
                } else if (alpha > 1) {
                    alpha = 1;
                }
                p2 = interval(0.5+0.5*alpha, curve.vertex[i], curve.vertex[j]);
                p3 = interval(0.5+0.5*alpha, curve.vertex[k], curve.vertex[j]);
                curve.tag[j] = "CURVE";
                curve.c[3 * j + 0] = p2;
                curve.c[3 * j + 1] = p3;
                curve.c[3 * j + 2] = p4;
            }
            curve.alpha[j] = alpha;
            curve.beta[j] =  0.5;
        }
        curve.alphacurve = 1;
    }

    private int   opti_penalty(Path path, int i,int  j, Opti res, double opttolerance, double []convc, double []areac) {
        int  m = path.curve.n;
        Curve curve = path.curve;
        Point[]vertex = curve.vertex;
                int k, k1, k2;
                double conv;
                int i1;
               double area, alpha, d, d1, d2;
                Point p0, p1, p2, p3, pt;
               double  A, R, A1, A2, A3, A4,
                s, t;

        if (i==j) {
            return 1;
        }

        k = i;
        i1 = mod(i+1, m);
        k1 = mod(k+1, m);
        conv = convc[k1];
        if (conv == 0) {
            return 1;
        }
        d =  ddist(vertex[i], vertex[i1]);
        for (k=k1; k!=j; k=k1) {
            k1 = mod(k+1, m);
            k2 = mod(k+2, m);
            if (convc[k1] != conv) {
                return 1;
            }
            if (sign(cprod(vertex[i], vertex[i1], vertex[k1], vertex[k2])) !=
                    conv) {
                return 1;
            }
            if (iprod1(vertex[i], vertex[i1], vertex[k1], vertex[k2]) <
                    d * ddist(vertex[k1], vertex[k2]) * -0.999847695156) {
                return 1;
            }
        }

        p0 = curve.c[mod(i,m) * 3 + 2].copy();
        p1 = vertex[mod(i+1,m)].copy();
        p2 = vertex[mod(j,m)].copy();
        p3 = curve.c[mod(j,m) * 3 + 2].copy();

        area = areac[j] - areac[i];
        area -= dpara(vertex[0], curve.c[i * 3 + 2], curve.c[j * 3 + 2])/2;
        if (i>=j) {
            area += areac[m];
        }

        A1 = dpara(p0, p1, p2);
        A2 = dpara(p0, p1, p3);
        A3 = dpara(p0, p2, p3);

        A4 = A1+A3-A2;

        if (A2 == A1) {
            return 1;
        }

        t = A3/(A3-A4);
        s = A2/(A2-A1);
        A =  (A2 * t / 2.0);

        if (A == 0.0) {
            return 1;
        }

        R = area / A;
        alpha =  (2 - Math.sqrt(4 - R / 0.3));

        res.c[0] = interval(t * alpha, p0, p1);
        res.c[1] = interval(s * alpha, p3, p2);
        res.alpha = alpha;
        res.t = t;
        res.s = s;

        p1 = res.c[0].copy();
        p2 = res.c[1].copy();

        res.pen = 0;

        for (k=mod(i+1,m); k!=j; k=k1) {
            k1 = mod(k+1,m);
            t =  tangent(p0, p1, p2, p3, vertex[k], vertex[k1]);
            if (t<-0.5) {
                return 1;
            }
            pt = bezier(t, p0, p1, p2, p3);
            d =  ddist(vertex[k], vertex[k1]);
            if (d == 0.0) {
                return 1;
            }
            d1 = dpara(vertex[k], vertex[k1], pt) / d;
            if (Math.abs(d1) > opttolerance) {
                return 1;
            }
            if (iprod(vertex[k], vertex[k1], pt) < 0 ||
                    iprod(vertex[k1], vertex[k], pt) < 0) {
                return 1;
            }
            res.pen += d1 * d1;
        }

        for (k=i; k!=j; k=k1) {
            k1 = mod(k+1,m);
            t =  tangent(p0, p1, p2, p3, curve.c[k * 3 + 2], curve.c[k1 * 3 + 2]);
            if (t<-0.5) {
                return 1;
            }
            pt = bezier(t, p0, p1, p2, p3);
            d =  ddist(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2]);
            if (d == 0.0) {
                return 1;
            }
            d1 = dpara(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2], pt) / d;
            d2 = dpara(curve.c[k * 3 + 2], curve.c[k1 * 3 + 2], vertex[k1]) / d;
            d2 *= 0.75 * curve.alpha[k1];
            if (d2 < 0) {
                d1 = -d1;
                d2 = -d2;
            }
            if (d1 < d2 - opttolerance) {
                return 1;
            }
            if (d1 < d2) {
                res.pen += (d1 - d2) * (d1 - d2);
            }
        }

        return 0;
    }

    private void  optiCurve(Path path) {


        Curve curve = path.curve;int  m = curve.n;
        Point[]vert = curve.vertex;
                int  []pt = new int [m + 1];
                double []pen = new double[m + 1];
                int[] len = new int [m + 1];
                        Opti []opt = new Opti[m + 1];
               int  om;int i, j,r;
                Opti o = new Opti();Point p0;
                int i1, area;double alpha; Curve ocurve;
        double []s; double[]t;

        double  []convc = new double[m]; double []areac = new double [m + 1];

        for (i=0; i<m; i++) {
            if (curve.tag[i] == "CURVE") {
                convc[i] = sign(dpara(vert[mod(i-1,m)], vert[i], vert[mod(i+1,m)]));
            } else {
                convc[i] = 0;
            }
        }

        area = 0;
        areac[0] = 0;
        p0 = curve.vertex[0];
        for (i=0; i<m; i++) {
            i1 = mod(i+1, m);
            if (curve.tag[i1] == "CURVE") {
                alpha = curve.alpha[i1];
                area += 0.3 * alpha * (4-alpha) *
                        dpara(curve.c[i * 3 + 2], vert[i1], curve.c[i1 * 3 + 2])/2;
                area += dpara(p0, curve.c[i * 3 + 2], curve.c[i1 * 3 + 2])/2;
            }
            areac[i+1] = area;
        }

        pt[0] = -1;
        pen[0] = 0;
        len[0] = 0;


        for (j=1; j<=m; j++) {
            pt[j] = j-1;
            pen[j] = pen[j-1];
            len[j] = len[j-1]+1;

            for (i=j-2; i>=0; i--) {
                r = opti_penalty(path, i, mod(j,m), o,  info.opttolerance, convc,
                        areac);
                if (r>0) {
                    break;
                }
                if (len[j] > len[i]+1 ||
                        (len[j] == len[i]+1 && pen[j] > pen[i] + o.pen)) {
                    pt[j] = i;
                    pen[j] = pen[i] + o.pen;
                    len[j] = len[i] + 1;
                    opt[j] = o;
                    o = new Opti();
                }
            }
        }
        om = len[m];
         ocurve = new Curve(om);
        s = new double[om];
        t = new double[om];

        j = m;
        for (i=om-1; i>=0; i--) {
            if (pt[j]==j-1) {
                ocurve.tag[i]     = curve.tag[mod(j,m)];
                ocurve.c[i * 3 + 0]    = curve.c[mod(j,m) * 3 + 0];
                ocurve.c[i * 3 + 1]    = curve.c[mod(j,m) * 3 + 1];
                ocurve.c[i * 3 + 2]    = curve.c[mod(j,m) * 3 + 2];
                ocurve.vertex[i]  = curve.vertex[mod(j,m)];
                ocurve.alpha[i]   = curve.alpha[mod(j,m)];
                ocurve.alpha0[i]  = curve.alpha0[mod(j,m)];
                ocurve.beta[i]    = curve.beta[mod(j,m)];
                s[i] = t[i] =  1.0;
            } else {
                ocurve.tag[i] = "CURVE";
                ocurve.c[i * 3 + 0] = opt[j].c[0];
                ocurve.c[i * 3 + 1] = opt[j].c[1];
                ocurve.c[i * 3 + 2] = curve.c[mod(j,m) * 3 + 2];
                ocurve.vertex[i] = interval(opt[j].s, curve.c[mod(j,m) * 3 + 2],
                        vert[mod(j,m)]);
                ocurve.alpha[i] = opt[j].alpha;
                ocurve.alpha0[i] = opt[j].alpha;
                s[i] = opt[j].s;
                t[i] = opt[j].t;
            }
            j = pt[j];
        }

        for (i=0; i<om; i++) {
            i1 = mod(i+1,om);
            ocurve.beta[i] = s[i] / (s[i] + t[i1]);
        }
        ocurve.alphacurve = 1;
        path.curve = ocurve;
    }
    private void processPath() {

        for (int  i = 0; i < pathlist.size(); i++) {

            Path path = pathlist.get(i);
            if (inShape==false&&path.sign == '+')continue;
                calcSums(path);
            calcLon(path);
            bestPolygon(path);
            adjustVertices(path);

            if (path.sign == '-') {
                reverse(path);
            }
            smooth(path);
            if (info.optcurve) {
                optiCurve(path);
            }
        }

    }




    private Point findNext(Point point) {
        int i = (int) (bm1.w * point.y + point.x);
        while (i < bm1.size && bm1.data[i]!= 1) {
            i++;
        }
        if(i < bm1.size )
            return bm1.index(i);
        else  return null;
    }

    private boolean  majority(double x, double y) {
        int  i, a, ct;
        for (i = 2; i < 5; i++) {
            ct = 0;
            for (a = -i + 1; a <= i - 1; a++) {
                ct += bm1.at(x + a, y + i - 1) ? 1 : -1;
                ct += bm1.at(x + i - 1, y + a - 1) ? 1 : -1;
                ct += bm1.at(x + a - 1, y - i) ? 1 : -1;
                ct += bm1.at(x - i, y + a) ? 1 : -1;
            }
            if (ct > 0) {
                return true;
            } else if (ct < 0) {
                return false;
            }
        }
        return false;
    }

    private  Path findPath(Point point) {
         Path path = new Path();
            double x =  point.x,y =  point.y, dirx = 0, diry = 1, tmp;

        path.sign = bm.at(point.x, point.y) ? '+' : '-';

        while (true) {
            path.pt.add(new Point(x, y));
            if (x > path.maxX)
                path.maxX = x;
            if (x < path.minX)
                path.minX = x;
            if (y > path.maxY)
                path.maxY = y;
            if (y < path.minY)
                path.minY = y;
            path.len++;

            x += dirx;
            y += diry;
            path.area -= x * diry;

            if (x == point.x && y == point.y)
                break;

            boolean l = bm1.at(x + (dirx + diry - 1 ) / 2, y + (diry - dirx - 1) / 2);
            boolean r = bm1.at(x + (dirx - diry - 1) / 2, y + (diry + dirx - 1) / 2);

            if (r && !l) {
                if (info.turnpolicy.equals("right") ||
                        (info.turnpolicy.equals( "black") && path.sign=='+' ||
                        (info.turnpolicy.equals("white") && path.sign == '-') ||
                        (info.turnpolicy.equals("majority") && majority(x, y)) ||
                        (info.turnpolicy.equals("minority") && !majority(x, y)))){
                    tmp = dirx;
                    dirx = -diry;
                    diry = tmp;
                } else {
                    tmp = dirx;
                    dirx = diry;
                    diry = -tmp;
                }
            } else if (r) {
                tmp = dirx;
                dirx = -diry;
                diry = tmp;
            } else if (!l) {
                tmp = dirx;
                dirx = diry;
                diry = -tmp;
            }
        }
        return path;
    }

    private void xorPath(Path path){
        double  y1 = path.pt.get(0).y,
                len = path.len,
                 x, y, maxX, minY; int i, j;
        for (i = 1; i < len; i++) {
            x = path.pt.get(i).x;
            y = path.pt.get(i).y;
            if (y != y1) {
                minY = y1 < y ? y1 : y;
                maxX = path.maxX;
                for (j = (int) x; j < maxX; j++) {
                    bm1.flip(j, minY);
                }
                y1 = y;
            }
        }

    }

    private void bmToPathlist() {
        /**将图片转为数组*/
        bm1 = bm.copy();
        Point  currentPoint = new Point(0, 0);
        Path path=new Path();
        while ((currentPoint =findNext(currentPoint))!=null) {
            path = findPath(currentPoint);
            xorPath(path);
            if (path.area > info.turdsize) {
                pathlist.add(path);
            }
        }
    }
    private void loadBm(byte[] bytes, int width, int height) {
        bm = new BitMap(width, height);
        bm.data=bytes;
        info.isReady = true;
    }

}
