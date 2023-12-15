import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.graph.LambdaLayer;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;

public class GANModelJava {

    public static class LambdaLayerCustom extends LambdaLayer {
        public LambdaLayerCustom(Builder builder, int... inputShape) {
            super(builder, inputShape);
        }

        @Override
        public void setNIn(int i) {
            super.setNIn(i);
        }

        @Override
        public int getNIn() {
            return super.getNIn();
        }
    }

    public static class Generator {
        public static ComputationGraphConfiguration.GraphBuilder createGeneratorConfig(int noiseSize, int hiddenSize, int maxTrajLen) {
            return new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(org.deeplearning4j.nn.conf.Updater.ADAM)
                    .weightInit(WeightInit.XAVIER)
                    .graphBuilder()
                    .addInputs("noiseInput")
                    .addLayer("layer1", new DenseLayer.Builder()
                            .nIn(noiseSize)
                            .nOut(hiddenSize * maxTrajLen / 4)
                            .activation(Activation.RELU)
                            .build(), "noiseInput")
                    .addLayer("lambda", new LambdaLayerCustom.Builder().lambda("X -> X.reshape(-1, " + hiddenSize + ", " + maxTrajLen / 4 + ")").build(), "layer1")
                    .addLayer("convTrans1", new Convolution1DLayer.Builder()
                            .nIn(hiddenSize)
                            .nOut(hiddenSize)
                            .kernelSize(8)
                            .stride(2)
                            .padding(3)
                            .activation(Activation.RELU)
                            .build(), "lambda")
                    .addLayer("convTrans2", new Convolution1DLayer.Builder()
                            .nIn(hiddenSize)
                            .nOut(hiddenSize)
                            .kernelSize(8)
                            .stride(2)
                            .padding(3)
                            .activation(Activation.RELU)
                            .build(), "convTrans1")
                    .addLayer("conv1d", new Convolution1DLayer.Builder()
                            .nIn(hiddenSize)
                            .nOut(3)
                            .kernelSize(7)
                            .padding(3)
                            .activation(Activation.TANH)
                            .build(), "convTrans2")
                    .setOutputs("conv1d");
        }
    }

    public static class Discriminator {
        public static ComputationGraphConfiguration.GraphBuilder createDiscriminatorConfig(int arrayLength, int hiddenSize) {
            return new NeuralNetConfiguration.Builder()
                    .seed(123)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(org.deeplearning4j.nn.conf.Updater.ADAM)
                    .weightInit(WeightInit.XAVIER)
                    .graphBuilder()
                    .addInputs("input")
                    .addLayer("lambda", new LambdaLayerCustom.Builder().lambda("X -> X.squeeze(1)").build(), "input")
                    .addLayer("conv1d_1", new Convolution1DLayer.Builder()
                            .nIn(3)
                            .nOut(hiddenSize)
                            .kernelSize(7)
                            .stride(2)
                            .padding(3)
                            .activation(Activation.LEAKYRELU)
                            .build(), "lambda")
                    .addLayer("conv1d_2", new Convolution1DLayer.Builder()
                            .nIn(hiddenSize)
                            .nOut(hiddenSize)
                            .kernelSize(7)
                            .stride(2)
                            .padding(3)
                            .activation(Activation.LEAKYRELU)
                            .build(), "conv1d_1")
                    .addLayer("conv1d_3", new Convolution1DLayer.Builder()
                            .nIn(hiddenSize)
                            .nOut(hiddenSize)
                            .kernelSize(7)
                            .stride(2)
                            .padding(3)
                            .activation(Activation.LEAKYRELU)
                            .build(), "conv1d_2")
                    .addLayer("lambda2", new LambdaLayerCustom.Builder().lambda("X -> X.reshape(-1, " + hiddenSize * arrayLength / 8 + ")").build(), "conv1d_3")
                    .addLayer("dense1", new DenseLayer.Builder()
                            .nIn(hiddenSize * arrayLength / 8)
                            .nOut(1)
                            .activation(Activation.LEAKYRELU)
                            .build(), "lambda2")
                    .setOutputs("dense1");
        }
    }

    public static void main(String[] args) {
        int noiseSize = 32;
        int hiddenSize = 64;
        int maxTrajLen = 128;

        ComputationGraphConfiguration generatorConfig = Generator.createGeneratorConfig(noiseSize, hiddenSize, maxTrajLen);
        ComputationGraph generator = new org.deeplearning4j.nn.graph.ComputationGraph(generatorConfig);
        generator.init();

        int arrayLength = 128;
        ComputationGraphConfiguration discriminatorConfig = Discriminator.createDiscriminatorConfig(arrayLength, hiddenSize);
        ComputationGraph discriminator = new org.deeplearning4j.nn.graph.ComputationGraph(discriminatorConfig);
        discriminator.init();

        // Training code goes here
    }
}
