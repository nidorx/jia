package com.github.nidorx.jia.ga;

import com.github.nidorx.jia.ga.decoder.DecoderDnaLayers;
import com.github.nidorx.jia.ga.selection.SelectionElite;
import com.github.nidorx.jia.ga.selection.SelectionStochasticUniversalSampling;
import com.github.nidorx.jia.ga.storage.Info;
import com.github.nidorx.jia.ga.storage.Storage;
import com.github.nidorx.jia.util.Callback;
import com.github.nidorx.jia.util.MultiException;
import com.github.nidorx.jia.util.JiaUtils;
import com.github.nidorx.jia.mlp.Input;
import com.github.nidorx.jia.mlp.Network;
import com.github.nidorx.jia.mlp.Output;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;

/**
 * Algoritmo genético para MLP
 *
 * O algoritmo da GA deve ser implementado e possuir os detalhes de execução do GA
 *
 * O algoritmo deve ser implementado para funcionar em multithread
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public abstract class Algorithm {

    private static final Logger LOG = Logger.getLogger(Algorithm.class.getName());

    /**
     * For compute-intensive tasks, an Ncpu-processor system usually achieves optimum utilization with a thread pool of
     * Ncpu +1 threads.
     */
    private static final int POOL_SIZE = Runtime.getRuntime().availableProcessors() + 1;

    /**
     * Pool de threads
     */
    private static final ExecutorService POOL = Executors.newFixedThreadPool(POOL_SIZE);

    /**
     * Informações sobre a população atual
     */
    private Population population;

    /**
     * Estado atual de execução do GA
     */
    private State state = State.STOPPED;

    /**
     * Variavel de controle para identificar a quantidade de individuos da população atual que já foi executada
     */
    private int countExecuted;

    /**
     * Lista de erros ocorridos durante a execução do algoritmo
     */
    private List<Throwable> errors;

    /**
     * Lista de callbacks que serão invocados quando finalzar o processamento
     */
    private final List<Callback> stopCallbacks = new ArrayList<>();

    /**
     * O mapeamento das entradas da Rede Neural
     *
     * @return
     */
    public abstract String[] getInputNames();

    /**
     * O mapeamento das saídas da Rede Neural
     *
     * @return
     */
    public abstract String[] getOutputNames();

    /**
     * Obtém o mecanismo de persistencia deste GA
     *
     * @return
     */
    public abstract Storage getStorage();

    /**
     * A execução do algoritmo, deve ser implementado
     *
     * @param input Input da Rede Neural do Cromossomo sendo executado
     * @param output Output da Rede Neural do Cromossomo sendo executado
     * @return O fitness (aptidão) do Cromossomo executado
     * @throws java.lang.Throwable
     */
    public abstract double run(Input input, Output output) throws Throwable;

    /**
     * A probabilidade de cruzamento dos individuos do GA
     *
     * @return
     */
    public double getProbabilityCrossover() {
        return Chromosome.PC;
    }

    /**
     * A probabilidade de mutação dos indivíduos do GA
     *
     * @return
     */
    public double getProbabilityMutation() {
        return Mutation.PM;
    }

    /**
     * Obtém o tamanho da população definida para execução do algoritmo.
     *
     * O tamanho da população não é constante, pode variar com o tempo de execução do GA.
     *
     * O algoritmo do GA tentará manter o tamanho constante, mas poderá haver variações com o tempo de execução.
     *
     * @return
     */
    public int getPopulationSize() {
        return 50;
    }

    /**
     * Permite definir uma porção da população inicial usando uma heurística conhecida para o problema.
     *
     * Independente de quantos inidividuos for definido, o algoritmo de GA irá adicionar no mínimo a quantidade
     * referente a 30% do tamanho da população, afim de aumentar a diversidade e garantir melhores resultados.
     *
     * Exemplo:
     *
     * Tamanho desejado da população: 30
     *
     * Indivídos definidos no método: 25
     *
     * Indivíduos criados pelo GA: 30*0.3 = 10
     *
     * Tamanho da população inicial: 25 + 10 = 35
     *
     * IMPORTATE! Os itens duplicados serão substituidos pelo GA
     *
     * @return
     */
    public List<Individual> getInitialPopulation() {
        return null;
    }

    /**
     * Executa a seleção da população que irá compor a geração seguinte
     *
     * Além destes, o algoritmo da GA irá adiconar a quantidade referente a 30% do tamanho de população de forma
     * aleatória, afim de aumentar a diversidade e garantir melhores resultados.
     *
     * IMPORTATE! Os itens duplicados serão substituidos pelo GA
     *
     * @param actualPopulation A população atual
     * @return Os genes da proxima geração
     */
    public List<Individual> getNewGeneration(List<Individual> actualPopulation) {

        final int size = getPopulationSize();
        final int selectionSize = (int) (size * 0.7);
        final List<Individual> selection = new ArrayList<>();

        // ------------------------------- 
        // [Seleção]
        // -------------------------------
        // Elitismo
        selection.addAll(// Seleciona 2 ou 5% melhores para compor nova população
                new SelectionElite()
                        .select((int) Math.min(Math.max(2, selectionSize * 0.05), actualPopulation.size()), actualPopulation)
                        .stream()
                        .collect(Collectors.toList())
        );

        // Seleciona 4 ou 20% aleatorios (usando Stochastic Universal Sampling.) para compor nova população
        selection.addAll(new SelectionStochasticUniversalSampling()
                .select((int) Math.min(Math.max(4, selectionSize * 0.2), actualPopulation.size()), actualPopulation)
                .stream()
                .collect(Collectors.toList())
        );

        final String[] inputNames = getInputNames();
        final String[] outputNames = getOutputNames();

        // -------------------------------
        //  [Cruzamento]
        // -------------------------------
        // efetuar 6 ou 20% cruzamento de pais aleatórios
        for (int i = 0, j = selection.size() - 1, k = (int) Math.max(4, selectionSize * 0.2); i < k; i++) {
            final Individual dad = selection.get(JiaUtils.between(0, j));
            final Individual mom = selection.get(JiaUtils.between(0, j));
            selection.add(new Individual(Crossover.random(dad.chromosome, mom.chromosome), inputNames, outputNames));
        }

        // -------------------------------
        //  [Mutação]
        // -------------------------------
        //  gerar 2 clones mutantes dos 2 mais bem adaptados
        final Chromosome firstChromosome = selection.get(0).chromosome;
        final Chromosome secondChromosome = selection.get(1).chromosome;
        selection.add(new Individual(Mutation.mutate(firstChromosome), inputNames, outputNames));
        selection.add(new Individual(Mutation.mutate(firstChromosome), inputNames, outputNames));
        selection.add(new Individual(Mutation.mutate(secondChromosome), inputNames, outputNames));
        selection.add(new Individual(Mutation.mutate(secondChromosome), inputNames, outputNames));

        // Até obter o tamanho esperado, adiciona individuos mutantes na população
        while (selection.size() < selectionSize) {
            final Individual individual = selection.get(JiaUtils.between(0, selection.size() - 1));
            selection.add(new Individual(Mutation.mutate(individual.chromosome), inputNames, outputNames));
        }

        return selection;
    }

    /**
     * Solicita o carregamento da ultima execução do algoritmo
     *
     * @throws java.lang.Exception
     */
    public void load() throws Exception {
        if (state.equals(State.RUNNING) || state.equals(State.STOPPING)) {
            throw new Exception("O estado do GA não permite o carregamento de dados.");
        }

        final Info info = this.getStorage().load();
        if (info == null) {
            return;
        }

        final List<Individual> individuals = new ArrayList<>();

        for (int i = 0, j = info.fitness.length; i < j; i++) {

            // Obtém o DNA e Fitness de cada individuo na ultima execução
            double[] dna = info.population[i];
            final double fitnes = info.fitness[i];

            final Individual individual = new Individual(new Chromosome(dna), getInputNames(), getOutputNames());
            individual.setFitness(fitnes);

            individuals.add(individual);
        }

        // Faz o carregamento da população
        population = new Population(
                info.generation,
                this.getPopulationSize(),
                individuals,
                this.getInputNames(),
                this.getOutputNames()
        );
    }

    /**
     * Inicia o processamento
     *
     * @throws java.lang.InterruptedException
     */
    public void start() throws Exception {

        if (state.equals(State.RUNNING)) {
            return;
        }

        if (state.equals(State.STOPPING)) {
            throw new Exception("O estado do GA não permite a inicialização.");
        }

        state = State.RUNNING;

        executeGeneration();
    }

    /**
     * Pausa o processamento
     *
     * @param callback Executado quando a finalização do processo ocorrer. Se o estado atual é STOPPED, o callback é
     * invocado imediatamente
     */
    public void stop(Callback callback) {
        if (state.equals(State.STOPPED)) {
            callback.call();
        } else {
            state = State.STOPPING;
            stopCallbacks.add(callback);
        }

    }

    /**
     * Executa uma nova população
     */
    private void executeGeneration() {
        // a)[Início] Gere uma população aleatória de n cromossomas (soluções adequadas para o problema)
        // b)[Adequação] Avalie a adequação f(x) de cada cromossoma x da população
        // c)[Nova população] Crie uma nova população repetindo os passos seguintes até que a nova população esteja completa
        //      [Seleção] Selecione de acordo com sua adequação (melhor adequação, mais chances de ser selecionado) dois cromossomas para serem os pais
        //      [Cruzamento] Com a probabilidade de cruzamento cruze os pais para formar a nova geração. Se não realizar cruzamento, a nova geração será uma cópia exata dos pais.
        //      [Mutação] Com a probabilidade de mutação, altere os cromossomas da nova geração nos locus (posição nos cromossomas).
        //      [Aceitação] Coloque a nova descendência na nova população
        // d)[Substitua] Utilize a nova população gerada para a próxima rodada do algoritmo
        // e)[Teste] Se a condição final foi atingida, pare, e retorne a melhor solução da população atual
        // f)[Repita] Vá para o passo 2

        // Zera o contador de finalizados
        this.countExecuted = 0;
        this.errors = new ArrayList<>();

        // Se não existe população inicial:
        if (population == null) {
            // Gera uma população aleatória
            population = new Population(
                    0,
                    this.getPopulationSize(),
                    this.getInitialPopulation(),
                    this.getInputNames(),
                    this.getOutputNames()
            );
        } else {
            // Gera a nova população para testes
            // Cosidera que a população existente já está salva, portanto, já foi testada
            population = new Population(
                    population.generation + 1,
                    this.getPopulationSize(),
                    this.getNewGeneration(Arrays.asList(population.individuals)),
                    this.getInputNames(),
                    this.getOutputNames()
            );
        }

        for (final Individual individual : population.individuals) {
            // Executa os individuos, em paraleleo (~1 individuo por CPU)
            CompletableFuture
                    .supplyAsync(new IndividualRunner(this, individual), POOL)
                    .whenComplete((updated, error) -> {

                        this.countExecuted++;

                        if (error != null) {

                            this.errors.add(error);
                            LOG.log(Level.WARNING, "Erro inesperado na execução do indivídio do GA", error);

                        } else if (updated.getError() != null) {

                            this.errors.add(updated.getError());
                            LOG.log(Level.WARNING, "Erro inesperado na execução do indivídio do GA", updated.getError());

                        } else {

                            LOG.log(Level.INFO, String.format("Indivíduo do GA executado com sucesso: %s | fitness %.10f | tempo %s",
                                    updated,
                                    updated.getFitness(),
                                    JiaUtils.time(updated.getEnd() - updated.getStart())
                            ));
                        }

                        whenExecuteGenerationComplete();
                    });

            // Após finalizar a execução de todos os itens, gera uma nova população
        }
    }

    private void whenExecuteGenerationComplete() throws CompletionException {
        if (this.countExecuted == this.population.individuals.length) {
            // Todos os individuos dessa população já foram processados

            if (!errors.isEmpty()) {
                try {
                    // Quando existem erros, não salva os dados da população, finaliza a execução do GA
                    errors.forEach((ex) -> {
                        ex.printStackTrace();
                    });
                    throw new MultiException("Erros na execução do GA", errors);
                } catch (MultiException ex) {
                    throw new CompletionException(ex);
                }
            }

            try {
                persist();
            } catch (Exception ex) {
                System.out.println("Erro inesperado ao persistir os dados da geração testada");
                ex.printStackTrace();
            } 

            if (state.equals(State.STOPPING)) {
                // Informar sobre a solicitação de parada de execução
                state = State.STOPPED;

                stopCallbacks.forEach(callback -> {
                    callback.call();
                });

                // Remove os callbacks
                stopCallbacks.clear();
            } else {

                // Executa a nova população
                executeGeneration();
            }
        }
    }

    /**
     * Solicita a persistencia da execução
     */
    private void persist() throws Exception {

        // Periste a população testada
        double[] fitness = new double[population.individuals.length];
        double[][] dnas = new double[population.individuals.length][];
        for (int i = 0, j = population.individuals.length; i < j; i++) {
            Individual individual = population.individuals[i];
            dnas[i] = individual.chromosome.getDna();
            fitness[i] = individual.getFitness() == null ? Double.NEGATIVE_INFINITY : individual.getFitness();
        }

        this.getStorage().save(new Info(population.generation, dnas, fitness));
    }

    /**
     * Executa a rede neural de um cromossomo, retornando a informação do indivíduo
     *
     * @param chromosome
     * @return 
     */
    public Individual execute(Chromosome chromosome) {
        final Individual individual = new Individual(chromosome, getInputNames(), getOutputNames());
        try {
            // Transforma o cromossomo do indivíduo na Rede Neural
            final Network network = individual.getNetwork();

            individual.setStart(System.currentTimeMillis());

            // Resultado da execução do algoritmo
            final double fitness = run(network.input(), network.output());

            individual.setEnd(System.currentTimeMillis());

            individual.setFitness(fitness);
        } catch (Throwable ex) {
            // Adicionar o erro no log de execução
            individual.setError(ex);
        }

        return individual;
    }

    /**
     * Tarefa responsável pela execução de um individuo
     */
    private static final class IndividualRunner implements Supplier<Individual> {

        private final Algorithm algorithm;

        private final Individual individual;

        public IndividualRunner(Algorithm algorithm, Individual individual) {
            this.algorithm = algorithm;
            this.individual = individual;
        }

        @Override
        public Individual get() {
            try {
                // Transforma o cromossomo do indivíduo na Rede Neural
                final Network network = new Network(
                        DecoderDnaLayers.getInstance().decode(individual.chromosome.getDna()),
                        Arrays.asList(algorithm.getInputNames()),
                        Arrays.asList(algorithm.getOutputNames())
                );

                individual.setStart(System.currentTimeMillis());

                // Resultado da execução do algoritmo
                final double fitness = algorithm.run(network.input(), network.output());

                individual.setEnd(System.currentTimeMillis());

                individual.setFitness(fitness);
            } catch (Throwable ex) {
                // Adicionar o erro no log de execução
                individual.setError(ex);
            }

            return individual;
        }

    }

    public static enum State {
        /**
         * O algoritmo genético está em execução
         */
        RUNNING,
        /**
         * Foi solicitado a parada da execução do algoritmo.
         */
        STOPPING,
        /**
         * O GA não está sendo executado no momento
         */
        STOPPED
    }
}
