package com.github.nidorx.jia.mlp;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fábrica de redes neurais
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Factory {

    /**
     * Criação de um network totalmente personalizada, com neuronios e camadas previamente definidas
     *
     * Método útil para restauração de networks salvas
     *
     * @param layers
     * @return
     */
    public static Network build(List<List<Neuron>> layers) {
        if (layers.size() < 2) {
            throw new IllegalArgumentException("É necessário informar ao menos 2 camadas (Hidden e Output)");
        }
        String[] input = new String[layers.get(0).size()];
        String[] output = new String[layers.get(layers.size() - 1).size()];

        for (int i = 0, l = layers.get(0).size(); i < l; i++) {
            input[i] = "input_" + i;
        }
        for (int i = 0, l = layers.get(layers.size() - 1).size(); i < l; i++) {
            output[i] = "output_" + i;
        }

        final Layer[] networkLayers = new Layer[layers.size()];
        for (int i = 0, j = layers.size(); i < j; i++) {
            int prevLayerSize = i == 0 ? input.length : layers.get(i - 1).size();
            final List<Neuron> neurons = layers.get(i);
            neurons.forEach(neuron -> {
                if (neuron.weights.length != prevLayerSize) {
                    throw new IllegalArgumentException("A quantidade de pesos do Neuron é inválido");
                }
            });
            networkLayers[i] = new Layer(neurons.toArray(new Neuron[]{}));
        }

        return new Network(networkLayers, initializeInputNames(input), initializeOutputNames(output));
    }

    /**
     * Criação de um network com as layers informadas e a função de transferencia comum a todos os neuronios
     *
     * @param layers Dimenção das camadas, incluindo input, hidden e output layer
     * @param transfer
     * @return
     */
    public static Network build(int[] layers, Transfer transfer) {
        if (layers.length < 3) {
            throw new IllegalArgumentException("É necessário informar ao menos 3 camadas (Input, Hidden e Output)");
        }

        String[] input = new String[layers[0]];
        String[] output = new String[layers[layers.length - 1]];
        int[] hidden = new int[layers.length - 2];
        System.arraycopy(layers, 1, hidden, 0, layers.length - 2);

        for (int i = 0, l = layers[0]; i < l; i++) {
            input[i] = "input_" + i;
        }
        for (int i = 0, l = layers[layers.length - 1]; i < l; i++) {
            output[i] = "output_" + i;
        }

        return initialize(input, hidden, output, transfer);
    }

    /**
     * Inicializa uma MLP onde todos os neuronios possuem a função de ativação definida
     *
     * @param input O nome das entradas, facilita o desenvolvimento
     * @param hidden Dimensões das camadas ocultas
     * @param output Os nomes dos neurons de saída
     * @param transfer
     * @return
     */
    public static Network build(String[] input, int[] hidden, String[] output, Transfer transfer) {
        if (hidden.length < 1) {
            throw new IllegalArgumentException("É necessário informar ao menos 1 camada Hidden");
        }

        return initialize(input, hidden, output, transfer);
    }

    /**
     * Inicializa uma rede com input e output mapeado definindo para CADA CAMADA o tipo de função de ativação dos seus
     * neuronios
     *
     * @param input
     * @param hidden
     * @param hiddenTransfer
     * @param output
     * @param outputTransfer
     * @return
     */
    public static Network build(String[] input, int[] hidden, String[] output, Transfer hiddenTransfer, Transfer outputTransfer) {
        if (hidden.length < 1) {
            throw new IllegalArgumentException("É necessário informar ao menos 1 camada Hidden");
        }

        if (hiddenTransfer == null) {
            throw new IllegalArgumentException("A função de transferência das camdas Hidden é requerida");
        }

        if (outputTransfer == null) {
            throw new IllegalArgumentException("A função de transferência das camada Output é requerida");
        }

        // Hidden + Output
        int[] sizes = new int[hidden.length + 1];
        System.arraycopy(hidden, 0, sizes, 0, hidden.length);
        sizes[hidden.length] = output.length;
        final Layer[] networkLayers = new Layer[sizes.length];

        for (int i = 0, j = sizes.length; i < j; i++) {
            if (i == 0) {
                networkLayers[i] = new Layer(sizes[i], input.length, hiddenTransfer);
            } else if (i == j - 1) {
                networkLayers[i] = new Layer(sizes[i], sizes[i - 1], outputTransfer);
            } else {
                networkLayers[i] = new Layer(sizes[i], sizes[i - 1], hiddenTransfer);
            }
        }

        return new Network(networkLayers, initializeInputNames(input), initializeOutputNames(output));
    }

    /**
     * Inicializa uma rede com input e output mapeado definindo para CADA NEURONIO o tipo de função de ativação
     *
     * @param input
     * @param hidden
     * @param output
     * @return
     */
    public static Network build(String[] input, List<List<Transfer>> hidden, Map<String, Transfer> output) {
        if (hidden.size() < 1) {
            throw new IllegalArgumentException("É necessário informar ao menos 1 camada Hidden");
        }

        final Layer[] networkLayers = new Layer[hidden.size() + 1];
        for (int i = 0, j = hidden.size(); i < j; i++) {
            List<Transfer> transfers = hidden.get(i);
            if (transfers.size() < 1) {
                throw new IllegalArgumentException("É necessário existir ao menos 1 Neuron na camada");
            }
            int prevLayerSize = i == 0 ? input.length : hidden.get(i - 1).size();
            final Neuron[] neurons = new Neuron[transfers.size()];
            for (int k = 0, l = transfers.size(); k < l; k++) {
                final Transfer transfer = transfers.get(k);
                if (transfer == null) {
                    throw new IllegalArgumentException("A função de transferência é requerida");
                }
                neurons[k] = new Neuron(prevLayerSize, transfer);
            }
            networkLayers[i] = new Layer(neurons);
        }

        final Neuron[] outputNeurons = new Neuron[output.size()];
        int i = 0;
        int prevLayerSize = hidden.get(hidden.size() - 1).size();
        output.values().forEach(transfer -> {
            outputNeurons[i] = new Neuron(prevLayerSize, transfer);
        });
        networkLayers[hidden.size()] = new Layer(outputNeurons);

        return new Network(
                networkLayers,
                initializeInputNames(input),
                initializeOutputNames(new ArrayList<>(output.keySet()).toArray(new String[]{}))
        );
    }

    private static Network initialize(String[] input, int[] hidden, String[] output, Transfer transfer) {
        if (transfer == null) {
            throw new IllegalArgumentException("A função de transferência é requerida");
        }

        // Hidden + Output
        int[] sizes = new int[hidden.length + 1];
        System.arraycopy(hidden, 0, sizes, 0, hidden.length);
        sizes[hidden.length] = output.length;

        final Layer[] networkLayers = new Layer[sizes.length];

        for (int i = 0; i < sizes.length; i++) {
            if (i == 0) {
                networkLayers[i] = new Layer(sizes[i], input.length, transfer);
            } else {
                networkLayers[i] = new Layer(sizes[i], sizes[i - 1], transfer);
            }
        }

        return new Network(networkLayers, initializeInputNames(input), initializeOutputNames(output));
    }

    /**
     * Define os nomes das camadas de entrada e saída
     *
     * Os nomes dos inputs e outputs são necessários para facilitar o mapeamento de valores de entrada e saida
     *
     * @param input
     * @param output
     */
    private static List<String> initializeInputNames(String[] input) {
        if (input.length < 1) {
            throw new IllegalArgumentException("É necessário informar ao menos 1 Input");
        }

        List<String> inputNames = new ArrayList<>(input.length);
        for (int i = 0, l = input.length; i < l; i++) {
            inputNames.add(i, input[i]);
        }

        return inputNames;
    }

    private static List<String> initializeOutputNames(String[] output) {

        if (output.length < 1) {
            throw new IllegalArgumentException("É necessário informar ao menos 1 Output");
        }

        List<String> outputNames = new ArrayList<>(output.length);
        for (int i = 0; i < output.length; i++) {
            outputNames.add(i, output[i]);
        }

        return outputNames;
    }
}
