package com.github.nidorx.jia.ga.storage;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Implementação para persistir as informações das gerações em diretório no disco
 *
 * Salva relatorio de evolução + dados da ultima geração
 *
 * Útil para testes durante a processo de implementação de um GA
 *
 * @author Alex
 */
public class StorageFile implements Storage {

    private final File dir;

    /**
     *
     * @param path Caminho para o diretório onde os dados serão persistidos
     */
    public StorageFile(String path) {
        this.dir = new File(path);
        if (!dir.exists()) {
            dir.mkdir();
        }
    }

    @Override
    public Info load() throws Exception {
        final Path path = dir.toPath().resolve("generation");
        if (!Files.exists(path)) {
            return null;
        }
        Integer generation = null;
        List<Double> fitness = new ArrayList<>();
        List<double[]> population = new ArrayList<>();
        try (Stream<String> stream = Files.lines(path)) {
            List<String> lines = stream.collect(Collectors.toList());
            for (int i = 0, j = lines.size(); i < j; i++) {
                final String line = lines.get(i).trim();
                if (i == 0) {
                    // Numero da geração
                    generation = Integer.valueOf(line);
                } else {
                    String[] parts = line.split(", ");
                    List<Double> individual = new ArrayList<>();
                    for (int k = 0, l = parts.length; k < l; k++) {
                        Double value = Double.valueOf(parts[k].trim());
                        if (k == 0) {
                            // fitness
                            fitness.add(value);
                        } else {
                            individual.add(value);
                        }
                    }
                    population.add(individual.stream().mapToDouble(n -> n).toArray());
                }
            }
        }

        double[][] poparr = new double[population.size()][];
        for (int i = 0, l = population.size(); i < l; i++) {
            poparr[i] = population.get(i);
        }
        return new Info(
                generation,
                poparr,
                fitness.stream().mapToDouble(i -> i).toArray()
        );
    }

    @Override
    public void save(Info info) throws Exception {
        try (PrintWriter pw = new PrintWriter(dir.toPath().resolve("generation").toFile())) {
            pw.println(info.generation);
            for (int i = 0, j = info.fitness.length; i < j; i++) {
                writeArrayChromossome(pw, info.fitness[i], info.population[i]);
            }
        }

        try (FileWriter w = new FileWriter(dir.toPath().resolve("evolution").toFile(), true)) {
            final String line = Arrays.toString(new double[]{
                info.bestFitness(), info.avgFitness(), info.worstFitness()
            });
            w.write(info.generation + ", ");
            w.write(line.substring(1, line.length() - 1) + "\n");
        }
    }

    private void writeArrayChromossome(final PrintWriter pw, double fitness, double[] dna) {
        final String line = Arrays.toString(dna);
        pw.print(String.valueOf(fitness) + ", ");
        pw.println(line.substring(1, line.length() - 1));
    }

    public static void main(String[] args) throws Exception {
        StorageFile strg = new StorageFile("C:\\dev\\projetos\\jia\\src\\com\\github\\nidorx\\jia\\ga\\storage\\tste");
        strg.save(
                new Info(
                        0,
                        new double[][]{
                            {Math.random(), Math.random()},
                            {Math.random(), Math.random()},
                            {Math.random(), Math.random()}
                        },
                        new double[]{Math.random(), Math.random(), Math.random()}
                )
        );

        Info loaded = strg.load();
        System.out.println(loaded);
    }

}
