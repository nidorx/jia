package com.github.nidorx.jia.ga.selection;

import com.github.nidorx.jia.ga.Individual;
import java.util.List;

/**
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public interface Selection {

    /**
     * Seleciona um número de cromossomos de uma populaçao
     *
     * @param number
     * @param population
     * @return
     */
    public List<Individual> select(int number, List<Individual> population);
}
