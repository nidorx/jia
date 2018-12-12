package com.github.nidorx.jia.ga.selection;

import com.github.nidorx.jia.ga.Individual;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

/**
 * Roulette Wheel Selection
 *
 * No método de seleção de Roleta [Holland, 1992], o primeiro passo é calcular a aptidão cumulativa de toda a população
 * através da soma da aptidão de todos os indivíduos. Depois disso, a probabilidade de seleção é calculada para cada
 * indivíduo.
 *
 * Então, uma matriz é construída contendo probabilidades cumulativas dos indivíduos. Assim, n números aleatórios são
 * gerados no intervalo 0 para a soma da aptidão. E para cada número aleatório é procurado um elemento de matriz que
 * pode ter maior valor. Portanto, os indivíduos são selecionados de acordo com suas probabilidades de seleção.
 *
 * Fonte:
 * https://github.com/giacomelli/GeneticSharp/blob/master/src/GeneticSharp.Domain/Selections/RouletteWheelSelection.cs
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class SelectionRouletteWheel implements Selection {

    @Override
    public List<Individual> select(int number, List<Individual> population) {
        if (number < 2) {
            throw new IllegalArgumentException("O número de indivíduos a selecionar deve ser ao menos 2");
        }

        if (population.size() < number) {
            throw new IllegalArgumentException("O número de indivíduos disponíveis deve se ao menos o tamanho da população");
        }

        double[] rouletteWheel = calculateCumulativePercentFitness(population);
        return selectFromWheel(number, population, rouletteWheel, () -> Math.random());
    }

    /**
     * Calculates the cumulative percent.
     *
     * @param population
     * @return
     */
    protected double[] calculateCumulativePercentFitness(List<Individual> population) {

        double[] rouletteWheel = new double[population.size()];
        double sumFitness = population.stream().mapToDouble(i -> i.getFitness()).sum();

        double cumulativePercent = 0.0;

        for (int i = 0, j = population.size(); i < j; i++) {
            cumulativePercent += population.get(i).getFitness() / sumFitness;
            rouletteWheel[i] = cumulativePercent;
        }

        return rouletteWheel;
    }

    /**
     * Selects from wheel.
     *
     * @param number
     * @param population
     * @param rouletteWheel
     * @param pointer
     * @return
     */
    protected List<Individual> selectFromWheel(
            int number,
            List<Individual> population,
            double[] rouletteWheel,
            Supplier<Double> pointer
    ) {
        List<Individual> selected = new ArrayList<>();

        for (int i = 0; i < number; i++) {
            double pointerValue = pointer.get();
            // First or selected
            int index = 0;
            for (int j = 0, k = rouletteWheel.length; j < k; j++) {
                if (rouletteWheel[j] >= pointerValue) {
                    index = j;
                    break;
                }
            }

            selected.add(population.get(index));
        }

        return selected;
    }

}
