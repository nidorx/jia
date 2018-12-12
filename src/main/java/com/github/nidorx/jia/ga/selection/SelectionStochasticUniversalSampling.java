package com.github.nidorx.jia.ga.selection;

import com.github.nidorx.jia.ga.Individual;
import java.util.List;
import java.util.concurrent.atomic.DoubleAccumulator;

/**
 * Stochastic Universal Sampling.
 *
 * A Amostragem Universal Estocástica é uma elaborada variação da Roleta.
 *
 * A Amostragem Universal Estocástica assegura que as freqüências de seleção observadas de cada indivíduo estejam em
 * linha com as freqüências esperadas. Então, se temos um indivíduo que ocupa 4,5% da roda e selecionamos 100
 * indivíduos, esperamos em média que esse indivíduo seja selecionado entre quatro e cinco vezes. A estocástica de
 * amostragem universal garante isso. O indivíduo será selecionado quatro vezes ou cinco vezes, não três vezes, nem zero
 * e nem 100 vezes.
 *
 * A seleção padrão Roleta não faz esta garantia.
 *
 * Fonte:
 * https://github.com/giacomelli/GeneticSharp/blob/master/src/GeneticSharp.Domain/Selections/StochasticUniversalSamplingSelection.cs
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class SelectionStochasticUniversalSampling extends SelectionRouletteWheel {

    @Override
    public List<Individual> select(int number, List<Individual> population) {

        if (population.size() < number) {
            throw new IllegalArgumentException("O número de indivíduos disponíveis deve se ao menos o tamanho da população");
        }

        double[] rouletteWheel = calculateCumulativePercentFitness(population);

        final DoubleAccumulator pointer = new DoubleAccumulator((x, y) -> x + y, Math.random());
        double stepSize = 1.0 / number;
        return selectFromWheel(number, population, rouletteWheel, () -> {
            if (pointer.get() > 1.0) {
                pointer.accumulate(-1.0);
            }

            double current = pointer.get();
            pointer.accumulate(stepSize);

            return current;
        });
    }
}
