package com.github.nidorx.jia.ga.storage;

/**
 * Informações sobre o ponto de execução do GA
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Info {

    /**
     * O número da geração desta população
     */
    public final int generation;

    /**
     * Os dnas de cada individuo desta população
     */
    public final double[][] population;

    /**
     * Os fitness de cada individuo desta população
     */
    public final double[] fitness;

    public Info(int generation, double[][] population, double[] fitness) throws Exception {
        this.generation = generation;
        this.population = population;
        this.fitness = fitness;
        if (population.length != fitness.length) {
            throw new Exception(
                    "A quantidade informação sobre a aptidão da população não correponde com a quantidade de individuos"
            );
        }
    }

    /**
     * Obtém a melhor aptidão dessa população
     *
     * @return
     */
    public double bestFitness() {
        double best = Double.NEGATIVE_INFINITY;

        for (double current : fitness) {
            best = current > best ? current : best;
        }

        return best;
    }

    /**
     * Obtém a aptidão média desta população
     *
     * @return
     */
    public double avgFitness() {
        double acc = 0.0;

        for (double current : fitness) {
            acc += current;
        }

        return acc / fitness.length;
    }

    /**
     * Obtém a pior aptidão desta população
     *
     * @return
     */
    public double worstFitness() {
        Double worst = Double.POSITIVE_INFINITY;

        for (double current : fitness) {
            worst = current < worst ? current : worst;
        }

        return worst;
    }

}
