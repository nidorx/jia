package com.github.nidorx.jia.ga.selection;

import com.github.nidorx.jia.ga.Individual;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Seleciona cromossomos com a melhor aptidão (Truncation Selection)
 *
 * Fonte:
 * https://github.com/giacomelli/GeneticSharp/blob/master/src/GeneticSharp.Domain/Selections/RouletteWheelSelection.cs
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class SelectionElite implements Selection {

    @Override
    public List<Individual> select(int number, List<Individual> population) {
        //        if (population.size() < 2) {;
        //            throw new IllegalArgumentException("O número de indivíduos disponíveis deve ser ao menos 2");
        //        }

        if (population.size() < number) {
            throw new IllegalArgumentException("O número de indivíduos disponíveis deve se ao menos o tamanho da população");
        }
        return population.stream()
                .sorted((a, b) -> a.getFitness().compareTo(b.getFitness()))
                .limit(number)
                .collect(Collectors.toList());
    }

}
