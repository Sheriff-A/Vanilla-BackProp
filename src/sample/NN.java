package sample;

import java.util.ArrayList;

public class NN {

    private ArrayList<double[]> hiddenLayers;
    private ArrayList<double[][]> weights; //weights.size() - 1; the last weight in the list is for the last hidden layer to the output layer
    private ArrayList<double[]> biases; //biases.size() - 1; the last bias in the list is for the last hidden layer to the output layer
    private double[] output;
    private final Function func;
    private final double learningRate;

    public NN(int input, int hidden, int out, int numLayers, double lr, int seed){
        System.out.println("---------- INIT ----------");

        // CREATION
        hiddenLayers = new ArrayList<double[]>();
        weights = new ArrayList<double[][]>();
        biases = new ArrayList<double[]>();
        learningRate = lr;
        func = new Function(seed);

        // INIT

        // HIDDEN LAYER FROM INPUT
        hiddenLayers.add(new double[hidden]);

        // WEIGHTS FROM INPUT
        weights.add(func.init2DArr(1, -0.5, hidden, input));

        // BIASES FROM INPUT
        biases.add(func.initArr(1, -0.5, hidden));

        for (int i = 1; i < numLayers; i++) {
            // HIDDEN LAYERS BETWEEN HIDDEN LAYERS
            hiddenLayers.add(new double[hidden]);

            // WEIGHTS BETWEEN HIDDEN LAYERS
            weights.add(func.init2DArr(1, -0.5, hidden, hidden));

            // BIASES BETWEEN HIDDEN LAYERS
            biases.add(func.initArr(1, -0.5, hidden));
        }

        // WEIGHTS TO OUTPUT
        weights.add(func.init2DArr(1, -0.5, out, hidden));

        // BIASES TO OUTPUT
        biases.add(func.initArr(1, -0.5, out));

        // OUTPUT LAYER
        output = new double[out];

        // PRINT INIT WEIGHTS AND BIASES
        for (int i = 0; i < weights.size(); i++) {
            // PRINT WEIGHTS
            System.out.println("----- WEIGHTS " + (i + 1) + " -----");
            func.print2DArr(weights.get(i));
            System.out.println();

            // PRINT BIASES
            System.out.println("----- BIASES " + (i + 1) + " -----");
            func.printArr(biases.get(i));
            System.out.println();
        }


    }

    public void train(int epochs, ArrayList<Input> training_data){
        System.out.println("---------- TRAINING ----------");

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("----- Epoch " + (epoch + 1) + " -----");
            func.shuffle(training_data);
            for (int i = 0; i < training_data.size(); i++) {
//            for (int i = 0; i < 1; i++) {
                Input input = training_data.get(i);

                // FEED FOWARD FROM INPUT
                feedFoward(input.getData(), hiddenLayers.get(0), weights.get(0), biases.get(0));

                // FEED FORWARD BETWEEN HIDDEN LAYERS
                for (int j = 1; j < hiddenLayers.size(); j++) {
                    feedFoward(hiddenLayers.get(j - 1), hiddenLayers.get(j), weights.get(j), biases.get(j) );
                }

                // FEED FORWARD TO OUTPUT
                feedFoward(hiddenLayers.get(hiddenLayers.size() - 1), output, weights.get(weights.size() - 1), biases.get(biases.size() - 1));

                // PRINT MSE EVERY N EPOCHS
                if (i % 1000 == 0) System.out.println("MSE: " + func.mse(output, input.getResult()));

                // BACK PROPAGATE THE RESULTS
                backProp(input);

            }
            // TRAINING ACCURACY
            System.out.println("TRAINING ACCURACY: " + test(training_data) + "%");
            System.out.println();
        }

        System.out.println("---------- DONE TRAINING ----------");

        // PRINT INIT WEIGHTS AND BIASES
        for (int i = 0; i < weights.size(); i++) {
            // PRINT WEIGHTS
            System.out.println("----- WEIGHTS " + (i + 1) + " -----");
            func.print2DArr(weights.get(i));
            System.out.println();

            // PRINT BIASES
            System.out.println("----- BIASES " + (i + 1) + " -----");
            func.printArr(biases.get(i));
            System.out.println();
        }
    }

    public double test(ArrayList<Input> testing_data){
        double accuracy = 0.0;
        for (Input input: testing_data) {
            feedFoward(input.getData(), hiddenLayers.get(0), weights.get(0), biases.get(0));

            // FEED FORWARD BETWEEN HIDDEN LAYERS
            for (int j = 1; j < hiddenLayers.size(); j++) {
                feedFoward(hiddenLayers.get(j - 1), hiddenLayers.get(j), weights.get(j), biases.get(j) );
            }

            // FEED FORWARD TO OUTPUT
            feedFoward(hiddenLayers.get(hiddenLayers.size() - 1), output, weights.get(weights.size() - 1), biases.get(biases.size() - 1));

            // Compare
            accuracy += (nnChoice(output) == input.getResult()) ? 1 : 0;
        }
        System.out.println();
        return (accuracy/testing_data.size()) * 100;
    }

    public void feedFoward(double[] from, double[] to, double[][] weight, double[] bias){
        for (int i = 0; i < to.length ; i++) {
            to[i] = bias[i];
            for (int j = 0; j < from.length; j++) {
                to[i] += from[j] * weight[i][j];
            }
            to[i] = func.sigmoid(to[i]);
        }
    }

    public void backProp(Input input){
        // CALCULATE LOSSES
        double[] outputLoss = func.loss(output, input.getResult());
        ArrayList<double[]> hiddenLayerLosses = new ArrayList<double[]>();

        // LOSS TO OUTPUT
        hiddenLayerLosses.add(func.loss(hiddenLayers.get(hiddenLayers.size() - 1), outputLoss, weights.get(weights.size() - 1)));

        // LOSSES TO HIDDEN LAYERS
        for (int j = hiddenLayers.size() - 2; j >= 0; j--) {
            hiddenLayerLosses.add(func.loss(hiddenLayers.get(j),  hiddenLayerLosses.get(hiddenLayerLosses.size() - 1), weights.get(j + 1)));
        }

        // UPDATE THE WEIGHTS AND BIASES TO OUTPUT
        update(weights.get(weights.size() - 1), func.deltaWeight(learningRate, hiddenLayers.get(hiddenLayers.size() - 1), outputLoss));
        update(biases.get(biases.size() - 1), func.deltaBias(learningRate, outputLoss));

        // UPDATE THE WEIGHTS AND BIASES TO HIDDEN LAYERS
        for (int j = 1; j < hiddenLayers.size() - 1; j++) {
            update(weights.get(j), func.deltaWeight(learningRate, hiddenLayers.get(j), hiddenLayerLosses.get(hiddenLayerLosses.size() - j - 1)));
            update(biases.get(j), func.deltaBias(learningRate, hiddenLayerLosses.get(hiddenLayerLosses.size() - j - 1)));
        }

        // UPDATE THE WEIGHTS AND BIASES FROM THE INPUT
        update(weights.get(0), func.deltaWeight(learningRate, input.getData(), hiddenLayerLosses.get(hiddenLayerLosses.size() - 1)) );
        update(biases.get(0), func.deltaBias(learningRate, hiddenLayerLosses.get(hiddenLayerLosses.size() - 1)) );
    }

    public int nnChoice(double[] output){
        int choice = -1;
        double max = Double.MIN_VALUE;
        for (int i = 0; i < output.length; i++) {
            if(max < output[i]){
                choice = i;
                max = output[i];
            }
        }
        return  choice;
    }

    public void update(double[][] weight, double[][] change){
        for (int i = 0; i < weight.length; i++) {
            for (int j = 0; j < weight[i].length; j++) {
                weight[i][j] += change[i][j];
            }
        }
    }

    public void update(double[] bias, double[] change){
        for (int i = 0; i < bias.length; i++) {
            bias[i] += change[i];
        }
    }


}
