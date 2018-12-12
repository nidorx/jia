package com.github.nidorx.jia.ga;

import com.github.nidorx.jia.util.Util;
import static com.github.nidorx.jia.ga.Chromosome.addLayer;
import static com.github.nidorx.jia.ga.Chromosome.changeLayerSize;
import static com.github.nidorx.jia.ga.Chromosome.countLayers;
import static com.github.nidorx.jia.ga.Chromosome.removeLayer;

/**
 * Classe especializada na mutação de Cromossomos
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Mutation {

    /**
     * Probabilidade de Mutação
     */
    public static final double PM = 0.1;

    /**
     * Retorna um novo Cromosomo contendo os genes do DNA atual com mutações aleatórias
     *
     * De forma aleatória, um número pequeno ( +- 25% do valor atual ) é adicionado ou subtraido dos valores do neuronio
     * [SLOPE, BIAS e WEIGH].
     *
     * Pode ocorrer a inclusão ou remoçõa de Neurons das camadas bem como a inclusão ou remoção de camadas inteiras.
     *
     * A função de transferencia [TYPE] dos neuronios nunca é alterada
     *
     * @param parent
     * @return
     */
    public static Chromosome mutate(Chromosome parent) {
        // MUTAÇÕES
        // Mapeamento e Alteração aleatoria de todos os <SLOPE>, <BIAS> e <WEIGHT>
        // Remoção/Inclusão aleatória de NEURONS nas camadas (e respectivos ajustes)
        // Remoção/Inclusão aleatória de CAMADAS (e respectivos ajustes)

        double[] dnaNew = parent.forEachNeuron(neuron -> {
            // Alteração aleatoria de todos os  <BIAS> e <WEIGHT>
            neuron.bias = Util.coin(PM) ? hard(neuron.bias) : soft(neuron.bias);

            // Mutação aleatória nos pesos
            for (int i = 0, l = neuron.weights.length; i < l; i++) {
                neuron.weights[i] = Util.coin(PM) ? hard(neuron.weights[i]) : soft(neuron.weights[i]);
            }
            return true;
        });

        // Obtém detalhes das camadas
        int[] layers = countLayers(dnaNew);

        // Remoção/Inclusão aleatória de NEURONS nas camadas (e respectivos ajustes)
        // Ignora camada de saída
        for (int i = 0, l = layers.length - 1; i < l; i++) {
            // Altera a quantidade de neurons da camada?
            if (!Util.coin(PM)) {
                continue;
            }

            int size = layers[i];
            int newSize = Util.between(size / 2, size + size / 2);
            if (size != newSize) {
                dnaNew = changeLayerSize(dnaNew, i, newSize);
            }
        }

        // obtém informação atualizada
        layers = countLayers(dnaNew);

        // Remoção/Inclusão aleatória de CAMADAS (e respectivos ajustes)
        for (int i = 1, idxActual = i, sizeActual = layers.length, l = layers.length - 1; i < l; i++, idxActual++) {
            // Adiciona ou remove camada?

            if (Util.coin(PM)) {
                // Adiciona

                // Aleatório entre a metade da menor e dobro da maior camada (ATUAL E SEGUINTE)
                int min = Math.min(layers[i], layers[i + 1]);
                int max = Math.max(layers[i], layers[i + 1]);
                int size = Util.between(min / 2, max + max / 2);

                dnaNew = addLayer(dnaNew, idxActual, size);

                idxActual++;
                sizeActual++;
                break;
            }

            if (Util.coin(PM)) {
                // Remove
                if (sizeActual < 2) {
                    // Deve existir no mínimo um Hidden e o Output Layer
                    continue;
                }
                dnaNew = removeLayer(dnaNew, idxActual);
                sizeActual--;
                idxActual--;
            }
        }

        return new Chromosome(dnaNew);
    }

    /**
     * De forma aleatória, um número pequeno ( +- 15% do valor atual ) é adicionado ou subtraido dos valores inicial
     *
     * @param input
     * @return
     */
    public static double soft(double input) {
        return Util.coin(PM) ? Util.between(input * .85, input * 1.15) : input;
    }

    /**
     * De forma aleatória, um número relativamente alto ( +- 40% do valor atual ) é adicionado ou subtraido dos valores
     * inicial
     *
     * @param input
     * @return
     */
    public static double hard(double input) {
        return Util.coin(PM) ? Util.between(input * .6, input * 1.4) : input;
    }

}
