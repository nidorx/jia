package com.github.nidorx.jia.ga;

import com.github.nidorx.jia.ga.decoder.DecoderDnaList;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;

public class CrossoverTest {

    private static final DecoderDnaList DECODER_DNA_LIST = DecoderDnaList.getInstance();

    public CrossoverTest() {
    }

    @Test
    public void testSinglePoint() {
        Chromosome dad = generate(1d, 3, new int[]{1}, 2);
        Chromosome mom = generate(2d, 3, new int[]{1}, 2);

        Chromosome result = Crossover.singlePoint(dad, mom);
        System.out.println(result);
    }

    @Test
    public void testTwoPoints() {
        Chromosome dad = generate(1d, 1, new int[]{1, 1, 1, 1, 1, 1, 1, 1}, 1);
        Chromosome mom = generate(2d, 1, new int[]{1, 1, 1, 1, 1, 1, 1, 1}, 1);

        Chromosome result = Crossover.twoPoints(dad, mom);
        System.out.println(result);
    }

    // Gera um cromossomo com os valores informados
    /**
     * Gera um cromossomo que representa uma rede com os valores informados
     *
     * @param value Valor para todos os itens dos neuronios
     * @param inputs Numero de inputs da Rede Neural
     * @param hidden Camadas ocultas da Rede Neural
     * @param outputs Numero de saídas da Rede Neural
     * @return
     */
    private Chromosome generate(double value, int inputs, int[] hidden, int outputs) {
        List<List<List<Double>>> data = new ArrayList<>();

        // Insere o output layer nos layers
        int[] layers = new int[hidden.length + 1];
        System.arraycopy(hidden, 0, layers, 0, hidden.length);
        layers[layers.length - 1] = outputs;

        for (int li = 0, l = layers.length; li < l; li++) {
            List<List<Double>> layer = new ArrayList<>();
            for (int j = 0, k = layers[li]; j < k; j++) {
                final List<Double> neuron = new ArrayList<>();

                // <TYPE>
                neuron.add(1d);
                // <SLOPE>
                neuron.add(value);
                // <BIAS>
                neuron.add(value);

                // WEIGHT
                int prev = li == 0 ? inputs : layers[li - 1];
                for (int i = 0; i < prev; i++) {
                    neuron.add(value);
                }

                layer.add(neuron);
            }
            data.add(layer);
        }

        return new Chromosome(DECODER_DNA_LIST.encode(data));
    }

}
