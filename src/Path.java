import java.util.ArrayList;
import java.util.List;

public class Path {
    public  double area = 0;
    public  int len = 0;
    public Curve curve;
    public List <Point>pt = new ArrayList<Point>();
    public  double minX = 100000;
    public  double minY = 100000;
    public  double maxX= -1;
    public  double maxY = -1;
    public  char sign='+';

    public double x0;
    public double y0;

    public List<Sum> sums;
    public double[]lon;
    public int []po;
    public int m;
}
