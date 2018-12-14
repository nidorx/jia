package com.github.nidorx.jia.mlp;

import com.github.nidorx.jia.util.JiaUtils;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Classe utilitaria usada para recuperação da saída de uma Network
 *
 * Sempre que for acionada, o output tras os dados mais recentes da network
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Output {

    private final Network network;

    private final Layer layer;

    private final List<String> names;

    public Output(Network network, Layer outputLayer, List<String> names) {
        this.network = network;
        this.layer = outputLayer;
        this.names = names;
    }

    /**
     * Obtém o valor da saída mapeado
     *
     * @return
     */
    public Map<String, Double> asMap() {
        final Map<String, Double> out = new HashMap<>();
        names.forEach(name -> out.put(name, this.layer.neurons[names.indexOf(name)].output));
        return out;
    }

    /**
     * Obtém o valor da saída remapeado para um range específico
     *
     * @see info.alexrodin.Util#unremap(double, double, double)
     * @param targetMin lower bound of the value's target range
     * @param targetMax upper bound of the value's target range
     * @return
     */
    public Map<String, Double> asMap(double targetMin, double targetMax) {
        final Map<String, Double> out = new HashMap<>();
        names.forEach(name -> {
            out.put(name, JiaUtils.unremap(this.layer.neurons[names.indexOf(name)].output, targetMin, targetMax));
        });
        return out;
    }

    /**
     * Obtém o valor de saída atual do Network como array de valores
     *
     * @return
     */
    public double[] asArray() {
        return Arrays.asList(this.layer.neurons).stream()
                .mapToDouble(neuron -> neuron.output)
                .toArray();
    }

    /**
     * Obtém o valor de saída atual do Network como array de valores, remapeado para um range específico
     *
     * @see info.alexrodin.Util#unremap(double, double, double)
     * @param targetMin lower bound of the value's target range
     * @param targetMax upper bound of the value's target range
     * @return
     */
    public double[] asArray(double targetMin, double targetMax) {
        return Arrays.asList(this.layer.neurons).stream()
                .mapToDouble(neuron -> JiaUtils.unremap(neuron.output, targetMin, targetMax))
                .toArray();
    }

    /**
     * Obtém o valor de saída atual do Network como lista de valores
     *
     * @return
     */
    public List<Double> asList() {
        return Arrays.asList(this.layer.neurons).stream()
                .map(neuron -> neuron.output)
                .collect(Collectors.toList());
    }

    /**
     * Obtém o valor de saída atual do Network como lista de valores, remapeado para um range específico
     *
     * @see info.alexrodin.Util#unremap(double, double, double)
     * @param targetMin lower bound of the value's target range
     * @param targetMax upper bound of the value's target range
     * @return
     */
    public List<Double> asList(double targetMin, double targetMax) {
        return Arrays.asList(this.layer.neurons).stream()
                .map(neuron -> JiaUtils.unremap(neuron.output, targetMin, targetMax))
                .collect(Collectors.toList());
    }

    /**
     * Obtém o valor de saída de todos os Neuronios da Rede Neural
     *
     * @return
     */
    public double[][] fromAllLayers() {
        final List<double[]> layers = new ArrayList<>();
        network.each((lr, li) -> {
            double[] neurons = new double[lr.size];
            lr.each((neu, ni) -> {
                neurons[ni] = neu.output;
            });
            layers.add(neurons);
        });

        double[][] out = new double[layers.size()][];
        for (int i = 0, j = layers.size(); i < j; i++) {
            out[i] = layers.get(i);
        }
        return out;
    }

    /**
     * Obtém o valor de saída de todos os Neuronios da Rede Neural, remapeado para um range específico
     *
     * @see info.alexrodin.Util#unremap(double, double, double)
     * @param targetMin lower bound of the value's target range
     * @param targetMax upper bound of the value's target range
     * @return
     */
    public double[][] fromAllLayers(double targetMin, double targetMax) {
        final List<double[]> layers = new ArrayList<>();
        network.each((lr, li) -> {
            double[] neurons = new double[lr.size];
            lr.each((neu, ni) -> {
                neurons[ni] = JiaUtils.unremap(neu.output, targetMin, targetMax);
            });
            layers.add(neurons);
        });
        double[][] out = new double[layers.size()][];
        for (int i = 0, j = layers.size(); i < j; i++) {
            out[i] = layers.get(i);
        }
        return out;
    }
}
