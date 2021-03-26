package sample;

public class Input {

    private final double[] data;
    private final int result;

    public Input(double[] d, int res){
        data = d;
        result = res;
    }

    public double[] getData(){
        return data;
    }

    public int getResult() {
        return result;
    }
}
