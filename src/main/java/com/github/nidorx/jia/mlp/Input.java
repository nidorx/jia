package com.github.nidorx.jia.mlp;

import com.github.nidorx.jia.util.JiaUtils;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/**
 * Padronização do processo de entrada de dados na Rede Neural
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Input {

    private final Network network;

    private final Layer layer;

    public final List<String> names;

    private final double[] values;

    private final Consumer propagate;

    public Input(Network network, Layer outputLayer, List<String> names, Consumer<double[]> propagate) {
        this.network = network;
        this.propagate = propagate;
        this.layer = outputLayer;
        this.names = names;
        this.values = new double[names.size()];
    }

    /**
     * Aplica a entrada de dados na Rede Neural
     *
     * @param input
     * @return
     */
    public Output set(double[] input) {
        System.arraycopy(input, 0, values, 0, input.length);
        propagate.accept(input);
        return network.output();
    }

    /**
     * Aplica a entrada de dados na Rede Neural
     *
     * Re-mapeia todos os números que estão no range informado para o range 0-1
     *
     * @param input
     * @param currentMin
     * @param currentMax
     * @return
     */
    public Output set(double[] input, double currentMin, double currentMax) {
        return set(Arrays.stream(input).map(d -> JiaUtils.remap(d, currentMin, currentMax)).toArray());
    }

    /**
     * Aplica a entrada de dados na Rede Neural
     *
     * @param input
     * @return
     */
    public Output set(final List<Double> input) {
        return set(input.stream().mapToDouble(d -> d).toArray());
    }

    /**
     * Aplica a entrada de dados na Rede Neural
     *
     * Re-mapeia todos os números que estão no range informado para o range 0-1
     *
     * @param input
     * @param currentMin
     * @param currentMax
     * @return
     */
    public Output set(List<Double> input, double currentMin, double currentMax) {
        return set(input.stream().mapToDouble(d -> JiaUtils.remap(d, currentMin, currentMax)).toArray());
    }

    /**
     * Aplica a entrada de dados na Rede Neural
     *
     * @param input
     * @return
     */
    public Output set(Map<String, Double> input) {
        return set(names.stream().mapToDouble(name -> input.get(name)).toArray());
    }

    /**
     * Aplica a entrada de dados na Rede Neural
     *
     * Re-mapeia todos os números que estão no range informado para o range 0-1
     *
     * @param input
     * @param currentMin
     * @param currentMax
     * @return
     */
    public Output set(Map<String, Double> input, double currentMin, double currentMax) {
        return set(names.stream().mapToDouble(name -> JiaUtils.remap(input.get(name), currentMin, currentMax)).toArray());
    }

    /**
     * Obtém os valores de entrada atuais da Rede Neural como arrays
     *
     * @return
     */
    public double[] get() {
        return Arrays.stream(values).map(d -> d).toArray();
    }

    /**
     * Obtém os valores de entrada atuais da Rede Neural como array, remapeado para um range específico
     *
     * @see info.alexrodin.Util#unremap(double, double, double)
     * @param targetMin lower bound of the value's target range
     * @param targetMax upper bound of the value's target range
     * @return
     */
    public double[] get(double targetMin, double targetMax) {
        return Arrays.stream(values).map(d -> JiaUtils.unremap(d, targetMin, targetMax)).toArray();
    }

}
