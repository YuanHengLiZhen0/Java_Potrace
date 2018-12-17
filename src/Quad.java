public class Quad {
     double []data = new double[]{0,0,0,0,0,0,0,0,0};

   public double at (int x, int y) {
        return this.data[x * 3 + y];
    };
}
