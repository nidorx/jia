package com.github.nidorx.jia.mlp;

import com.github.nidorx.jia.util.Util;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.function.BiConsumer;

/**
 * Representação básica de uma A Rede Neural Artificial (Multi-layer Perceptron)
 *
 * Utilize o input para entrada de dados na Rede Neural (Feedforward Propagation)
 *
 * Utilize o output para obter o resultado da execução da Rede Neural
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Network {

    /**
     * Enumerador dos algortimos implementados disponíveis para treinar a MLP
     */
    public static enum LEARNING_METHOD {
        STOCHASTIC_GRADIENT_DESCENT,
        BATCH_GRADIENT_DESCENT,
        MINI_BATCH_GRADIENT_DESCENT
    }

    /**
     * O Training Learning Rate padrão da rede
     */
    private static final double LEARNING_RATE = 0.03;

    /**
     * As camadas deste MLP
     */
    private final Layer[] layers;

    /**
     * Utilitário que permite inserir dados de entrada na Network
     */
    private final Input input;

    /**
     * Utilitário de acesso aos valores de saída do Network
     */
    private final Output output;

    /**
     * Training Learning Rate
     *
     * A taxa de aprendizagem controla o quanto mudar o peso para corrigir o erro.
     *
     * Por exemplo, um valor de 0.1 atualizará o peso de 10% do valor que possivelmente poderia ser atualizado.
     *
     * As pequenas taxas de aprendizagem são preferidas que causam uma aprendizagem mais lenta em um grande número de
     * iterações de treinamento. Isso aumenta a probabilidade de a rede encontrar um bom conjunto de pesos em todas as
     * camadas em vez do conjunto mais rápido de pesos que minimizam o erro (chamado de convergência prematura).
     */
    private double learningRate = LEARNING_RATE;
    
    private LEARNING_METHOD learningMethod = LEARNING_METHOD.STOCHASTIC_GRADIENT_DESCENT;

    public Network(Layer[] layers, String[] inputNames, String[] outputNames) {
        this(layers, Arrays.asList(inputNames), Arrays.asList(outputNames));
    }

    /**
     * Criação da MLP (Multi-layer Perceptron)
     *
     * @param layers
     * @param inputNames
     * @param outputNames
     */
    public Network(Layer[] layers, List<String> inputNames, List<String> outputNames) {
        this.layers = layers;
        this.input = new Input(this, this.layers[0], inputNames, row -> forwardPropagate(row));
        this.output = new Output(this, this.layers[this.layers.length - 1], outputNames);
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public Input input() {
        return input;
    }

    /**
     * Permite acesso aos dados de saída da Network.
     *
     *
     * @return
     */
    public Output output() {
        return output;
    }

    /**
     * Utilitário para navegar em todos os layers do Network
     *
     * @param fn
     */
    public void each(BiConsumer<Layer, Integer> fn) {
        for (int i = 0, l = layers.length; i < l; i++) {
            fn.accept(layers[i], i);
        }
    }

    /**
     * Limpa os dados temporários de todos os neuronios da rede
     */
    public void clear() {
        each((layer, i) -> {
            layer.clear();
        });
    }

    /**
     * Imprime o grafico com os valores de saída de cada neuronio
     *
     * @return
     */
    public String printGraph() {
        return printGraph(input.get(), output.fromAllLayers());
    }

    /**
     * Imprime o grafico com os valores de saída de cada neuronio, remapeado para um range específico
     *
     * @see info.alexrodin.Util#unremap(double, double, double)
     * @param targetMin lower bound of the value's target range
     * @param targetMax upper bound of the value's target range
     * @return
     */
    public String printGraph(double targetMin, double targetMax) {
        return printGraph(input.get(targetMin, targetMax), output.fromAllLayers(targetMin, targetMax));
    }

    private String printGraph(double[] inputs, double[][] outputs) {
        StringBuilder sb = new StringBuilder();
        final Locale L = java.util.Locale.US;

        // Hidden + Output + Input
        String[] lines = new String[layers.length + 1];

        // Input layer
        lines[0] = "";
        lines[0] = input.names.stream()
                .map(name -> "[" + String.format(L, "%.16f", inputs[input.names.indexOf(name)]) + "] ")
                .reduce(lines[0], String::concat);
        int maxLength = lines[0].length();

        // Ignore input layer (li = 1)
        for (int i = 0, l = outputs.length; i < l; i++) {
            final double[] layer = outputs[i];
            final int k = i + 1;
            lines[k] = "";
            for (int j = 0, m = layer.length; j < m; j++) {
                lines[k] = lines[k].concat("[" + String.format(L, "%.16f", layer[j]) + "] ");
            }
            lines[k] = lines[k].trim();
            maxLength = Math.max(maxLength, lines[k].length());
        }

        System.out.println("layer:");
        for (int i = 0, l = lines.length; i < l; i++) {
            String line = lines[i];
            if (i == 0) {
                sb.append("   in: ");
            } else if (i == l - 1) {
                sb.append("  out: ");
            } else {
                sb.append(String.format("% 4d : ", i));
            }
            for (int j = 0; j < (maxLength - line.length()) / 2; j++) {
                // Left pad
                sb.append(" ");
            }
            sb.append(line).append("\n");
        }

        return sb.toString();
    }

    /**
     * Obtém os detalhes das camadas e dos neurons da rede
     *
     * @return
     */
    public String printNeurons() {
        StringBuilder sb = new StringBuilder();
        final Locale L = java.util.Locale.US;

        // Imprimir no formato
        // ----------------------------
        // [ 0]  <SIZE: >
        //         ............. neuron 001
        // [ 0]    <TYPE  : >
        // [ 1]    <SLOPE : >
        // [ 2]    <BIAS  : >
        // [ 3]    <WEIGHT:> -- Peso para entrada [0...PREV]
        // ----------------------------
        each((layer, li) -> {
            if (li > 0) {
                sb.append(String.format(".................................\n\n"));
            }
            sb.append(String.format(L, ".. layer %03d ....................\n", li + 1));
            // Primeiro neuron da camada
            sb.append(String.format(L, "  <SIZE: %d>\n", layer.size));

            layer.each((neuron, ni) -> {
                sb.append(String.format(L, "    .................. neuron %03d\n", ni + 1));
                sb.append(String.format(L, "    <TYPE  : %s>\n", neuron.transfer.name));
                sb.append(String.format(L, "    <BIAS  : %+.16f>\n", neuron.bias));
                for (double weight : neuron.weights) {
                    sb.append(String.format(L, "    <WEIGHT: %+.16f>\n", weight));
                }
            });
        });

        sb.append(String.format(".................................\n"));
        return sb.toString();
    }

    /**
     * Imprime o netowork em dois formatos distintos, mapa e detalhamento
     *
     * @return
     */
    @Override
    public String toString() {
        return printGraph() + "\n" + printNeurons();
    }

    /**
     * Faz o treinamento da rede neural, usando stochastic gradient descent
     *
     * @param dataset Conjunto de treianmento
     * @param expecteds Valores esperados
     * @param maxError Percentagem de erro mínimo esperado (ex 0.05 para 95% de acerto mínimo)
     * @param epochs Quantidade máxima de épocas de treinamento
     * @throws java.lang.Exception Quando não consegue encontrar uma convergencia
     */
    public void train(double[][] dataset, double[][] expecteds, double maxError, int epochs) throws Exception {
        long start = System.currentTimeMillis();
        for (int epoch = 0; epoch < epochs; epoch++) {
            Double errorTotal = 0.0;

            if (learningMethod == LEARNING_METHOD.STOCHASTIC_GRADIENT_DESCENT) {
                for (int i = 0, l = dataset.length; i < l; i++) {
                    double[] row = dataset[i];
                    forwardPropagate(row);
                    double[] expected = expecteds[i];
                    double[] outputs = output.asArray();
                    for (int j = 0, k = expected.length; j < k; j++) {
                        final double error = expected[j] - outputs[j];
                        errorTotal += Math.pow(error, 2);
                    }
                    backPropagate(expected);
                    updateWeights(row);
                }
            }

            if (epoch % 1000 == 0) {
                // Log a cada 1000 epocas
                System.out.println(String.format("epoch: %d, error: %.3f", epoch, errorTotal));
            }

            if (errorTotal <= maxError) {
                // Atingiu a percentagem de acerto mínimo definida
                long end = System.currentTimeMillis();
                System.out.println("-----------------------------------------------------------");
                System.out.println(String.format("Treinamento finalizado em %s ", Util.time(end - start)));
                System.out.println(String.format("epoch: %d, error: %.5f", epoch, errorTotal));
                System.out.println("-----------------------------------------------------------");
                return;
            }
        }

        throw new Exception("Não foi possível encontrar uma convergência");
    }

    /**
     * Forward Propagation
     *
     * @param row
     */
    protected void forwardPropagate(double[] row) {

        double[] inputs = row;

        // Execute - hiddens + output
        for (Layer layer : layers) {
            final double[] inputs_p = inputs;
            final double[] outputs = new double[layer.size];
            layer.each((neuron, j) -> {
                outputs[j] = neuron.activate(inputs_p);
            });
            inputs = outputs;
        }

    }

    /**
     * backpropagate starting with the otuput layer working backwards
     *
     * @param expected the expected output value for the neuron
     */
    protected void backPropagate(double[] expected) {
        // Reversed
        for (int l = layers.length - 1, a = l; a >= 0; a--) {
            final Layer layer = layers[a];

            if (a == l) {
                // Calcula o erro e delta do output layer
                layer.each((neuron, j) -> {
                    double error = expected[j] - neuron.output;
                    neuron.gradient(error);
                });
            } else {
                // Hidden layers
                final int k = a;

                // Calcula o erro e delta da camada atual, usando o delta da camada seguinte
                layer.each((actual, i) -> {
                    final DoubleAdder error = new DoubleAdder();
                    layers[k + 1].each((next, j) -> {
                        error.add(next.delta * next.weights[i]);
                    });
                    actual.gradient(error.sum());
                });
            }
        }
    }

    /**
     * Update network weights with error
     *
     * updates the weights for a network given an input row of data, a learning rate and assume that a forward and
     * backward propagation have already been performed.
     *
     * @param row
     */
    protected void updateWeights(double[] row) {
        for (int a = 0, l = layers.length; a < l; a++) {
            final int i = a;
            final double[] inputs;

            if (i == 0) {
                // Input Layer
                inputs = row;
            } else {
                // Hidden and Output layer
                Layer layer = layers[i - 1];
                inputs = new double[layer.size];
                layer.each((neuron, j) -> {
                    inputs[j] = neuron.output;
                });
            }

            layers[i].each((neuron, b) -> {
                for (int j = 0, m = inputs.length; j < m; j++) {
                    neuron.weights[j] += learningRate * neuron.delta * inputs[j];
                }
                neuron.bias += learningRate * neuron.delta;
            });
        }
    }

}
