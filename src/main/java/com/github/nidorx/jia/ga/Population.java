package com.github.nidorx.jia.ga;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Population {

    /**
     * Informações sobre o número da geração desta população
     */
    public final int generation;

    public final Individual[] individuals;

    public Population(int generation, int size, List<Individual> initial, String[] inputNames, String[] outputNames) {

        final int inputSize = inputNames.length;
        final int outputSize = outputNames.length;

        // if (minSize < 2)throw new ArgumentOutOfRangeException("minSize", "The minimum size for a population is 2 chromosomes.");
        this.generation = generation;
        if (initial == null || initial.isEmpty()) {
            // Geração aleatória de todos os indivíduos
            individuals = new Individual[size];
            for (int i = 0; i < size; i++) {
                individuals[i] = new Individual(Chromosome.random(inputSize, outputSize), inputNames, outputNames);
            }
        } else {
            int numRandom = (int) Math.max(size - initial.size(), size * 0.30);
            int total = numRandom + initial.size();
            individuals = new Individual[total];

            for (int i = 0, j = initial.size(); i < j; i++) {
                final Individual individual = initial.get(i);

                // Validar se o chromossomo possui o tamanho de entrada e saida iguais ao informado
                int[] sizes = individual.chromosome.getLayersSizes();
                if (sizes[0] != inputSize) {
                    throw new IllegalArgumentException("O cromossomo possui número de entradas diferente do esperado");
                }

                if (sizes[sizes.length - 1] != outputSize) {
                    throw new IllegalArgumentException("O cromossomo possui número de saídas diferente do esperado");
                }

                // Mantém a referencia para os individuos já criados
                individuals[i] = individual;
            }

            // Finaliza a população com individuos aleatórios
            for (int i = 0, j = initial.size(); i < numRandom; i++, j++) {
                individuals[j] = new Individual(Chromosome.random(inputSize, outputSize), inputNames, outputNames);
            }
        }

        // Substitui qualquer duplicado por aleatório
        final Set<Individual> uniques = new HashSet<>(Arrays.asList(individuals));
        if (uniques.size() < individuals.length) {
            while (uniques.size() < individuals.length) {
                uniques.add(new Individual(Chromosome.random(inputSize, outputSize), inputNames, outputNames));
            }
            System.arraycopy(uniques.toArray(new Individual[]{}), 0, individuals, 0, individuals.length);
        }
    }

    /**
     * Obtém o indivíduo com maior aptidão
     *
     * @return
     */
    public Individual bestFitness() {
        Individual best = null;

        for (Individual current : individuals) {
            if (best == null) {
                best = current;
            } else {
                best = current.getFitness() > best.getFitness() ? current : best;
            }
        }

        return best;
    }

    public double avgFitness() {
        double acc = 0.0;

        for (Individual individual : individuals) {
            acc += individual.getFitness();
        }

        return acc / individuals.length;
    }

    public Individual worstFitness() {
        Individual worst = null;

        for (Individual current : individuals) {
            if (worst == null) {
                worst = current;
            } else {
                worst = current.getFitness() < worst.getFitness() ? current : worst;
            }
        }

        return worst;
    }
}
