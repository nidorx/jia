package com.github.nidorx.jia.ga.decoder;

import com.github.nidorx.jia.ga.Chromosome;
import java.util.ArrayList;
import java.util.List;

/**
 * Métodos genéricos para tranformação de dados
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class DecoderDnaList implements Decoder<List<List<List<Double>>>, double[]> {

    private DecoderDnaList() {
    }

    private static class SingletonHelper {

        private static final DecoderDnaList INSTANCE = new DecoderDnaList();
    }

    public static DecoderDnaList getInstance() {
        return SingletonHelper.INSTANCE;
    }

    /**
     * Converte uma estrutura de lista de neuronios em DNA
     *
     * Formato de entrada:
     *
     * [ [ [TYPE, BIAS, ...WEIGHT], [T, B, ...W] ], [ [T, B, ...W] ]]
     *
     * Formato de saida:
     *
     * [SIZE, PREV, TYPE, BIAS, ...WEIGHT, SZ, P, T, B, ...W , SZ, P, T, B, ...W]
     *
     * @param layers
     * @return
     */
    @Override
    public double[] encode(List<List<List<Double>>> layers) {
        List<Double> dna = new ArrayList<>();
        List<List<Double>> prev = null;
        for (List<List<Double>> layer : layers) {

            // <SIZE>
            dna.add(Double.valueOf(layer.size()));
            // <PREV>
            final Integer tprevSize;
            if (prev == null) {
                tprevSize = layer.get(0).size() - Chromosome.NEURON_FIELDS;
            } else {
                tprevSize = prev.size();
            }
            dna.add(Double.valueOf(tprevSize));

            layer.forEach((neuron) -> {

                int expectedSize = tprevSize + Chromosome.NEURON_FIELDS;
                if (neuron.size() != expectedSize) {
                    System.out.println("DNA inconsistente. Dados do Neuron com tamanho inválido para o numero de entradas da camada");
                }
                while (neuron.size() < expectedSize) {
                    // Repete o ultimo peso para todas as entradas
                    neuron.add(neuron.get(neuron.size() - 1));
                }
                while (neuron.size() > expectedSize) {
                    // Remove os pesos que estão sobrando
                    neuron.remove(neuron.size() - 1);
                }

                dna.addAll(neuron);
            });
        }
        return dna.stream().mapToDouble(d -> d).toArray();
    }

    /**
     * Converte o DNA numa estrutura de listas
     *
     * Formato de entrada:
     *
     * [SIZE, PREV, TYPE, BIAS, ...WEIGHT, SZ, P, T, B, ...W , SZ, P, T, B, ...W]
     *
     * Formato de saida:
     *
     * [ [ [TYPE, BIAS, ...WEIGHT], [T, B, ...W] ], [ [T, B, ...W] ]]
     *
     * @param dna
     * @return
     */
    @Override
    public List<List<List<Double>>> decode(double[] dna) {
        List<List<List<Double>>> layers = new ArrayList<>();
        List<List<Double>> actuaLayer = new ArrayList<>();
        Chromosome.forEachNeuron(dna, neuron -> {

            if (neuron.index == 0) {
                if (!actuaLayer.isEmpty()) {
                    layers.add(new ArrayList<>(actuaLayer));
                }
                actuaLayer.clear();
            }

            final List<Double> values = new ArrayList<>();

            values.add(Double.valueOf(neuron.type));
            values.add(neuron.bias);

            for (double weight : neuron.weights) {
                values.add(weight);
            }

            actuaLayer.add(values);

            return true;
        });

        layers.add(new ArrayList<>(actuaLayer));

        return layers;
    }

}
