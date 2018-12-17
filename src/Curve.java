public class Curve {
       public int n ;
        public String[]tag ;
       public Point[] c ;
        public int alphacurve;
    public Point [] vertex;
    public double[] alpha;
    public double[] alpha0 ;
    public double[] beta ;

    public Curve(int n) {
        this.n = n;
        this.alphacurve=0;
        this.tag=new String[n];
        this.c=new Point[n*3];
        this.vertex=new Point[n];
        this.alpha=new double[n];
        this.alpha0=new double[n];
        this.beta=new double[n];
    }
}
