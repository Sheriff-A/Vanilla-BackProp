package sample;

import java.io.File;
import java.util.ArrayList;
import java.util.Scanner;

public class Main{

    public ArrayList<Input> training_data;
    public ArrayList<Input> testing_data;
    public NN nn;

    public Main(){
        training_data = new ArrayList<Input>();
        testing_data = new ArrayList<Input>();
        loadDigits("digits", "digit_train_", ".txt", 10,"digit_test_", ".txt", 10);
        double[] parameters = getParameters();
        nn = new NN(training_data.get(0).getData().length,(int)parameters[0], 10, (int)parameters[1], parameters[2], (int)parameters[4]);
        nn.train((int)parameters[3], training_data);
        System.out.println("TEST ACCURACY: " + nn.test(testing_data) + "%");
    }

    public double[] getParameters(){
        /*
        * Parameters for the NN *(Defaults)*:
        * Hidden Layer Size (80)
        * Number of Hidden Layers (2)
        * Learning Rate (0.01)
        * Number of Epochs (20)
        * Seed (49)
        * */
        double[] params = new double[] {80, 2, 0.01, 20, 49};

        // User Input

        return params;
    }

    public void loadDigits(String main_path, String trainFileName, String trainFileType,  int numTrainFiles,
                         String testFileName, String testFileType, int numTestFiles){
        try{
            // Loading Training data
            for (int i = 0; i < numTrainFiles; i++) {
                File f = new File(main_path + "\\" + trainFileName + i + trainFileType);
                Scanner scan = new Scanner(f);
                while(scan.hasNext()){
                    String[] line = scan.nextLine().split(",");
                    Input input = new Input(stringToDoubleArr(line), i);
                    training_data.add(input);
                }
                scan.close();
            }

            // Loading Testing data
            for (int i = 0; i < numTestFiles; i++) {
                File f = new File(main_path + "\\" + testFileName + i + testFileType);
                Scanner scan = new Scanner(f);
                while(scan.hasNext()){
                    String[] line = scan.nextLine().split(",");
                    Input input = new Input(stringToDoubleArr(line), i);
                    testing_data.add(input);
                }
                scan.close();
            }
        }catch (Exception e){
            System.out.println(e);
        }

    }

    public double[] stringToDoubleArr(String[] str){
        double[] d = new double[str.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = Double.parseDouble(str[i]);
        }
        return d;
    }

    public static void main(String[] args){
        Main m = new Main();
    }
}