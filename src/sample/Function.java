package sample;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import static java.lang.Math.exp;
import static java.lang.Math.pow;

public class Function {

    private final Random rand;

    public Function(int seed){
        rand = new Random(seed);
    }

    public double[] initArr(double range, double offset, int size){
        double[] arr = new double[size];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = (rand.nextDouble() * range) + offset;
        }
        return arr;

    }

    public double[][] init2DArr(double range, double offset, int x, int y){
        double[][] arr = new double[x][y];
        for (int i = 0; i < arr.length; i++) {
            for (int j = 0; j < arr[i].length; j++) {
                arr[i][j] = (rand.nextDouble() * range) + offset;
            }

        }
        return arr;
    }

    public double sigmoid(double x){
        return 1/(1 + exp(-x));
    }

    public double sigmoidDerv(double x){
        return x * (1 - x);
    }

    public double mse(double[] output, int result){
        double[] expected = new double[output.length];
        expected[result] = 1;
        double sum = 0;
        for (int i = 0; i < output.length; i++) {
            sum += pow((output[i] - expected[i]), 2);
        }
        return sum/output.length;

    }

    public double[] loss(double[] output, int result){
        double[] res = new double[output.length];
        double[] expected = new double[output.length];
        expected[result] = 1;
        for (int i = 0; i < res.length; i++) {
            res[i] = sigmoidDerv(output[i]) * -(output[i] - expected[i]);
        }
        return res;
    }

    public double[] loss(double[] curr, double[] loss, double[][] weight){
        double[] res = new double[curr.length];
        for (int i = 0; i < curr.length; i++) {
            res[i] = sigmoidDerv(curr[i]);
            double sum = 0;
            for (int j = 0; j < loss.length; j++) {
                sum += loss[j] * weight[j][i];
            }
            res[i] *= sum;
        }
        return res;
    }

    public double[][] deltaWeight(double lr, double[]curr, double[] loss){
        double[][] res = new double[loss.length][curr.length];
        for (int i = 0; i < loss.length; i++) {
            for (int j = 0; j < curr.length; j++) {
                res[i][j] = lr * loss[i] * curr[j];
            }
        }
        return res;
    }

    public double[] deltaBias(double lr, double[] loss){
        double[] res = new double[loss.length];
        for (int i = 0; i < loss.length; i++) {
            res[i] = lr * loss[i];
        }
        return res;
    }

    public void printArr(double[] arr){
        for (double d: arr) {
            System.out.print(clamp(d, 3) + " \t");
        }
        System.out.println();
    }

    public void print2DArr(double[][] arr){
        for (double[] col: arr) {
            for (double row: col) {
                System.out.print(clamp(row, 3) + " \t");
            }
            System.out.println();
        }
    }

    public double clamp(double d, int dec){
        return ((double) ((int) (d * pow(10, dec))))/(pow(10, dec));
    }

    public void shuffle(ArrayList list){
        Collections.shuffle(list, rand);
    }
}
