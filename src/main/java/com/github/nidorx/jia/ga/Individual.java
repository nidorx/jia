package com.github.nidorx.jia.ga;

import com.github.nidorx.jia.ga.decoder.DecoderDnaLayers;
import com.github.nidorx.jia.mlp.Network;
import java.util.Objects;

/**
 * Representação de um índivíduo de uma população
 *
 * Um indivíduo possui um cromossomo e, após sua execução, um valor de aptidão (fitness)
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Individual {

    public final Chromosome chromosome;

    public final String[] inputNames;

    public final String[] outputNames;

    private Network network;

    /**
     * A aptidão do inidividuo, calculado após a execução do mesmo na resolução do problem
     */
    private Double fitness;

    /**
     * Momento do iníco de execução do individuo
     */
    private Long start;

    /**
     * Momento do fim da execução do individuo
     */
    private Long end;

    /**
     * Erro de execução, se houver
     */
    private Throwable error;

    public Individual(Chromosome chromosome, String[] inputNames, String[] outputNames) {
        this.chromosome = chromosome;
        this.inputNames = inputNames;
        this.outputNames = outputNames;
    }

    /**
     * Obtém a RNA do cromossomo do indivíduo
     *
     * @return
     */
    public Network getNetwork() {
        if (network == null) {
            network = new Network(
                    DecoderDnaLayers.getInstance().decode(chromosome.getDna()),
                    inputNames,
                    outputNames
            );
        }
        return network;
    }

    public Double getFitness() {
        if (fitness == null) {
            return Double.NEGATIVE_INFINITY;
        }
        return fitness;
    }

    public void setFitness(double fitness) {
        if (this.fitness == null) {
            this.fitness = fitness;
        }
    }

    public long getStart() {
        return start;
    }

    public void setStart(long start) {
        if (this.start == null) {
            this.start = start;
        }
    }

    public long getEnd() {
        return end;
    }

    public void setEnd(long end) {
        if (this.end == null) {
            this.end = end;
        }
    }

    public Throwable getError() {
        return error;
    }

    public void setError(Throwable error) {
        if (this.error == null) {
            this.error = error;
        }
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(this.chromosome);
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Individual other = (Individual) obj;
        return Objects.equals(this.chromosome, other.chromosome);
    }

}
