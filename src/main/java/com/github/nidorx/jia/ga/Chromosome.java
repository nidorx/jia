package com.github.nidorx.jia.ga;

import com.github.nidorx.jia.util.Util;
import com.github.nidorx.jia.mlp.Transfer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * Representação de uma Rede Neural dentro de um GA
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Chromosome {

    // Probabilidade de Crossover
    public static final double PC = 0.7;

    // Informações do layer <SIZE> e <PREV>
    public static final int LAYER_FIELDS = 2;

    // Informações do neuron <TYPE>, e <BIAS>
    public static final int NEURON_FIELDS = 2;

    // Índice para o atributo <SIZE> de um Layer
    public static final int IDX_L_SIZE = 0;

    // Índice para o atributo <PREV> de um Layer
    public static final int IDX_L_PREV = 1;

    // Índice para o atributo <TYPE> de um Neuron
    public static final int IDX_N_TYPE = 0;

    // Índice para o atributo <BIAS> de um Neuron
    public static final int IDX_N_BIAS = 1;

    // Índice para o atributo <WEIGHT> de um Neuron
    public static final int IDX_N_WEIGHT = 2;

    // Valor mínimo para o BIAS de um Neuron
    public static final double BIAS_MIN = 0.01;

    // Valor máximo para o BIAS de um Neuron
    public static final double BIAS_MAX = 1.0;

    // Valor mínimo para o WEIGHT de um Neuron
    public static final double WEIGHT_MIN = 0.0;

    // Valor máximo para o WEIGHT de um Neuron
    public static final double WEIGHT_MAX = 1.0;

    // ====================================
    // SIZE     = Tamanho (em neurons) de uma camada (hidden e outpt)
    // PREV     = Tamanho (em neurons) da camada anterior 
    // TYPE     = Tipo de funçao de transferencia
    // BIAS     = Bias do neuron
    // WEIGHT   = Peso de uma conexao ( repete-se o tamanho da entrada anterior
    // ------------------------------------
    // Esquema
    // ------------------------------------
    // [ 0]  <SIZE>
    // [ 1]  <PREV>
    //         -- Neuron [0...SIZE]
    // [ 0]    <TYPE>
    // [ 1]    <BIAS>
    // [ 2]    <WEIGHT> -- Peso para entrada [0...PREV]
    // ------------------------------------
    private final double[] dna;

    private String cachedToString;

    private int[] cachedLayersSizes;
    
    private int[] cachedCountedLayers;

    private List<double[]> cachedExtractedLayers;

    /**
     * Cache do hashcode, evida processamento desnecessário
     */
    private int hash = -1;

    public Chromosome(double[] dna) {
        this.dna = dna;
    }

    public double[] getDna() {
        double[] out = new double[dna.length];
        System.arraycopy(dna, 0, out, 0, dna.length);
        return out;
    }

    /**
     * Permite acessar de forma sequencial todos os neurons do DNA do cromossomo atual
     *
     * @param callback
     * @return
     */
    public double[] forEachNeuron(Function<Neuron, Boolean> callback) {
        // @TODO: Usar cached
        return forEachNeuron(dna, callback);
    }

    /**
     * Permite acessar de forma sequencial os neurons do DNA do cromossomo atual a partir do ponto determinado pelo
     * índice do layer e índice do neuron
     *
     * @param callback
     * @param layerIdx
     * @param neuronIdx
     * @return
     */
    public double[] forEachNeuron(Function<Neuron, Boolean> callback, int layerIdx, int neuronIdx) {
        // @TODO: Usar cached
        return forEachNeuron(dna, callback, layerIdx, neuronIdx);
    }

    /**
     * Obtém a quantidade de layers que o dna deste cromossomo possui
     *
     * @return
     */
    public int[] countLayers() {
        if (cachedCountedLayers == null) {
            cachedCountedLayers = countLayers(dna);
        }
        return cachedCountedLayers;
    }

    /**
     * Obtém os valores mapeados dos layers deste cromossomo
     *
     * @return
     */
    public List<double[]> extractLayers() {
        if (cachedExtractedLayers == null) {
            cachedExtractedLayers = extractLayers(dna);
        }
        return cachedExtractedLayers;
    }

    /**
     * Obtém a informação sobre o tamanho de todas as layers, incluindo Input, Hidden e Output
     *
     * @return
     */
    public int[] getLayersSizes() {
        if (cachedLayersSizes == null) {
            List<double[]> layers = extractLayers();
            // (Hidden + Output) + Input
            int[] sizes = new int[layers.size() + 1];
            for (int i = 0, j = layers.size(); i < j; i++) {
                if (i == 0) {
                    // Input size
                    sizes[0] = Double.valueOf(layers.get(i)[IDX_L_PREV]).intValue();
                }
                sizes[i + 1] = Double.valueOf(layers.get(i)[IDX_L_SIZE]).intValue();
            }
            cachedLayersSizes = sizes;
        }
        return cachedLayersSizes;
    }

    @Override
    public String toString() {
        if (cachedToString == null) {
            // Imprimir no formato
            // ----------------------------
            // [ 0]  <SIZE: >
            // [ 1]  <PREV: >
            //         ............. neuron 001
            // [ 0]    <TYPE  : >
            // [ 1]    <BIAS  : >
            // [ 2]    <WEIGHT:> -- Peso para entrada [0...PREV]
            // ----------------------------
            StringBuilder sb = new StringBuilder();
            AtomicInteger c = new AtomicInteger(0);
            final Locale L = java.util.Locale.US;
            forEachNeuron(neuron -> {
                int i = c.get();
                if (neuron.index == 0) {
                    if (neuron.layer > 0) {
                        sb.append(String.format(".................................\n\n"));
                    }
                    sb.append(String.format(L, ".. layer %03d ....................\n", neuron.layer + 1));
                    // Primeiro neuron da camada
                    sb.append(String.format(L, " [%04d]  <SIZE: %d>\n", i++, neuron.layerSize));
                    sb.append(String.format(L, " [%04d]  <PREV: %d>\n", i++, neuron.weights.length));
                }
                sb.append(String.format(L, "           ......................\n", neuron.index + 1));
                sb.append(String.format(L, "           .......... neuron %03d\n", neuron.index + 1));
                sb.append(String.format(L, " [%04d]    <TYPE  : %d>\n", i++, neuron.type));
                sb.append(String.format(L, " [%04d]    <BIAS  : %.10f>\n", i++, neuron.bias));
                for (double weight : neuron.weights) {
                    sb.append(String.format(L, " [%04d]    <WEIGHT: %.10f>\n", i++, weight));
                }

                c.set(i);
                return true;
            });

            sb.append(String.format(".................................\n"));
            cachedToString = sb.toString();
        }

        return cachedToString;
    }

    @Override
    public int hashCode() {
        if (hash == -1) {
            hash = 7;
            hash = 97 * hash + Arrays.hashCode(this.dna);
        }
        return hash;
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
        final Chromosome other = (Chromosome) obj;
        return Arrays.equals(this.dna, other.dna);
    }

    /**
     * Altera a quantidade de neuronios da camada informada e retorna o dna atualizado
     *
     * Se a nova quantidade for maior que a atual, cria neuronios aleatórios NO FINAL DA CAMADA ATUAL e atualiza os
     * pesos dos neuronios da camada seguinte
     *
     * Se a nova quantidade é menor, REMOVE NEURONIOS FINAIS DA CAMADA ATUAL e remove os respectivos pesos dos neuronios
     * da camada seguinte
     *
     * @param dna
     * @param layerIndex
     * @param newSize
     * @return
     */
    public static double[] changeLayerSize(double[] dna, int layerIndex, int newSize) {
        if (newSize == 0) {
            // Impede a remoção de todos os neuronios
            return dna;
        }

        final List<double[]> newLayers = new ArrayList<>();
        final List<double[]> layers = extractLayers(dna);

        int oldSize = 0;

        // ----------------
        // [ 0]  (2.0)  <SIZE>
        // [ 1]  (3.0)  <PREV>
        //                 -- Neuron [0]
        // [ 2]  (1.0)      <TYPE>
        // [ 3]  (1.0)      <BIAS>
        // [ 4]  (1.0)      <WEIGHT> [0]
        // [ 5]  (1.0)      <WEIGHT> [1]
        // [ 6]  (1.0)      <WEIGHT> [PREV]
        //                 -- Neuron [SIZE]
        // [ 7]  (1.0)      <TYPE>
        // [ 8]  (1.0)      <BIAS>
        // [ 9]  (1.0)      <WEIGHT> [0]
        // [10]  (1.0)      <WEIGHT> [1]
        // [11]  (1.0)      <WEIGHT> [PREV]
        // ----------------
        for (int actualIndex = 0, total = layers.size(); actualIndex < total; actualIndex++) {
            double[] layer = layers.get(actualIndex);

            // Alteração aleatoria de todos os <TYPE>, <SLOPE>, <BIAS> e <WEIGHT>
            // Numero de neurons na camada
            int size = ((Double) layer[IDX_L_SIZE]).intValue();

            // Numero de neurons na camada anterior (indica o numero de pesos para cada neuron)
            int inputs = ((Double) layer[IDX_L_PREV]).intValue();

            if (layerIndex == actualIndex) {

                if (actualIndex == total - 1) {
                    // Não permite modificar o output layer (ultima camada)
                    return dna;
                }

                oldSize = size;

                if (oldSize == newSize) {
                    // A camada já possui a quantidade de neuros solicitada, não é necessário processamento
                    return dna;
                }

                final List<Double> newLayer = DoubleStream.of(layer).boxed().collect(Collectors.toList());
                // Novo <SIZE>
                newLayer.set(0, Double.valueOf(newSize));

                if (newSize > oldSize) {
                    // Se a nova quantidade for maior que a atual, cria neuronios aleatórios NO FINAL DA CAMADA ATUAL
                    int diff = newSize - oldSize;
                    for (int c = 0, d = diff; c < d; c++) {
                        // <TYPE>, <BIAS>
                        newLayer.add(Neuron.TYPE.random());
                        newLayer.add(Util.between(BIAS_MIN, BIAS_MAX));
                        for (int e = 0; e < inputs; e++) {
                            // <WEIGHT>
                            newLayer.add(Util.between(WEIGHT_MIN, WEIGHT_MAX));
                        }
                    }
                } else {
                    // Se a nova quantidade é menor, REMOVE NEURONIOS FINAIS DA CAMADA ATUAL
                    int diff = oldSize - newSize;
                    int itensToRemove = diff * NEURON_FIELDS + diff * inputs;
                    for (int c = itensToRemove, d = newLayer.size() - 1; c > 0; c--, d--) {
                        newLayer.remove(d);
                    }
                }

                // Salva referencia
                layer = newLayer.stream().mapToDouble(d -> d).toArray();

            } else if (layerIndex + 1 == actualIndex) {
                layer = updateNextLayer(layer, oldSize, newSize);
            }

            newLayers.add(layer);
        }

        return newLayers.stream().flatMapToDouble(Arrays::stream).toArray();
    }

    /**
     * Remove o layer do indice informado
     *
     * @param dna
     * @param index
     * @return
     */
    public static double[] removeLayer(double[] dna, int index) {
        if (index == 0) {
            // Impede a remoção do Input Layer
            throw new IllegalArgumentException("Não é permitido remover a camada Iput Layer");
        }

        List<double[]> layers = extractLayers(dna);

        for (int actualIndex = 0, total = layers.size(); actualIndex < total; actualIndex++) {
            double[] actual = layers.get(actualIndex);
            if (index == actualIndex) {
                if (actualIndex == total - 1) {
                    throw new IllegalArgumentException("Não é permitido remover a camada output Layer");
                }

                // Remove a camada
                layers.remove(index);

                // Atualiza o layer seguinte                
                int oldSize = ((Double) actual[IDX_L_SIZE]).intValue();
                int size = ((Double) layers.get(index - 1)[IDX_L_SIZE]).intValue();
                layers.set(index, updateNextLayer(layers.get(index), oldSize, size));
                break;
            }
        }

        return layers.stream().flatMapToDouble(Arrays::stream).toArray();
    }

    /**
     * Adiciona um novo layer no indice informado
     *
     * @param dna
     * @param index
     * @param size
     * @return
     */
    public static double[] addLayer(double[] dna, int index, int size) {
        if (size == 0) {
            // Impede criação de camadas sem neuronios
            throw new IllegalArgumentException("Não é permitido a criação de camadas sem neuronios");
        }

        List<double[]> layers = extractLayers(dna);

        for (int actualIndex = 0, total = layers.size(); actualIndex < total; actualIndex++) {
            double[] actual = layers.get(actualIndex);
            if (index == actualIndex) {
                if (actualIndex == total - 1) {
                    throw new IllegalArgumentException("Não é permitido inserir camada após o Output Layer");
                }

                final List<Double> newLayer = new ArrayList<>();
                // <SIZE>
                newLayer.add(Double.valueOf(size));
                // <PREV> com o <SIZE> do layer modificado
                newLayer.add(actual[IDX_L_SIZE]);

                // Numero de neurons na camada anterior (indica o numero de pesos para cada neuron
                int inputs = ((Double) actual[IDX_L_SIZE]).intValue();
                for (int c = 0; c < size; c++) {
                    // <TYPE>, <BIAS>
                    newLayer.add(Neuron.TYPE.random());
                    newLayer.add(Util.between(BIAS_MIN, BIAS_MAX));
                    for (int e = 0; e < inputs; e++) {
                        // <WEIGHT>
                        newLayer.add(Util.between(WEIGHT_MIN, WEIGHT_MAX));
                    }
                }

                // Insere o novo layer
                layers.add(index + 1, newLayer.stream().mapToDouble(d -> d).toArray());

                // Atualiza o layer seguinte                
                int oldSize = ((Double) actual[IDX_L_SIZE]).intValue();
                layers.set(index + 2, updateNextLayer(layers.get(index + 2), oldSize, size));
                break;
            }
        }

        return layers.stream().flatMapToDouble(Arrays::stream).toArray();
    }

    /**
     * Atualiza os neuronios de uma camada quando a camada anterior sofrer alteração
     *
     * @param nextLayer
     * @param previousLayer
     * @param prevLayerOldSize
     * @return
     */
    private static double[] updateNextLayer(double[] nextLayer, int prevLayerOldSize, int prevLayerNewSize) {
        // Camada SEGUINTE a camada modificada

        // Numero de neurons na camada
        int size = ((Double) nextLayer[IDX_L_SIZE]).intValue();

        // Numero de neurons na camada anterior (indica o numero de pesos para cada neuron)
        int inputs = ((Double) nextLayer[IDX_L_PREV]).intValue();

        final List<Double> newLayer = new ArrayList<>();

        // <SIZE>
        newLayer.add(nextLayer[IDX_L_SIZE]);

        // <PREV> com o <SIZE> do layer modificado
        newLayer.add(Double.valueOf(prevLayerNewSize));

        // Inicia o neuron com o índice do <TYPE>
        int startIndex = LAYER_FIELDS + IDX_N_TYPE;

        // Fator de incremento de neuron, representa o número de valores disponiveis para cópia, incluindo
        // os fields do neuron e os pesos das entradas
        int numValues = inputs + NEURON_FIELDS;

        // Para cada neuronio desta camada
        for (int c = 0; c < size; startIndex += numValues, c++) {
            for (int d = startIndex, k = startIndex + numValues; d < k; d++) {
                // <TYPE>, <BIAS> e <WEIGHT> do layer atual
                newLayer.add(nextLayer[d]);
            }

            if (prevLayerNewSize > prevLayerOldSize) {
                // Se a nova quantidade for maior que a atual, atualiza os pesos dos neuronios da camada seguinte
                // Adicionar os novos pesos para os novos neuronios
                int diff = prevLayerNewSize - prevLayerOldSize;
                for (int e = 0, f = diff; e < f; e++) {
                    // <WEIGHT>
                    newLayer.add(Util.between(WEIGHT_MIN, WEIGHT_MAX));
                }
            } else {
                // Se a nova quantidade é menor, remove os respectivos pesos dos neuronios da camada seguinte
                // Remover os pesos dos neuronios removidos
                int diff = prevLayerOldSize - prevLayerNewSize;
                for (int g = diff, h = newLayer.size() - 1; g > 0; g--, h--) {
                    newLayer.remove(h);
                }
            }
        }

        // Retorna o layer atualizado
        return newLayer.stream().mapToDouble(d -> d).toArray();
    }

    public static double[] forEachNeuron(double[] dna, Function<Neuron, Boolean> callback) {
        return forEachNeuron(dna, callback, 0, 0);
    }

    /**
     * Permite acessar de forma sequencial e mapeada todos os neurons do DNA informado
     *
     * @param dna A cadeia de DNA a ser percorrida
     * @param callback
     * @param layerIdx
     * @param neuronIdx
     * @return
     */
    public static double[] forEachNeuron(double[] dna, Function<Neuron, Boolean> callback, int layerIdx, int neuronIdx) {

        final Neuron neuron = new Neuron();
        final List<double[]> layers = extractLayers(dna);
        boolean stop = false;
        // ----------------
        // [ 0]  (2.0)  <SIZE>
        // [ 1]  (3.0)  <PREV>
        //                 -- Neuron [0]
        // [ 2]  (1.0)      <TYPE>
        // [ 3]  (1.0)      <BIAS>
        // [ 4]  (1.0)      <WEIGHT> [0]
        // [ 5]  (1.0)      <WEIGHT> [1]
        // [ 6]  (1.0)      <WEIGHT> [PREV]
        //                 -- Neuron [SIZE]
        // [ 7]  (1.0)      <TYPE>
        // [ 8]  (1.0)      <BIAS>
        // [ 9]  (1.0)      <WEIGHT> [0]
        // [10]  (1.0)      <WEIGHT> [1]
        // [11]  (1.0)      <WEIGHT> [PREV]
        // ----------------
        for (int li = 0, lsz = layers.size(); li < lsz; li++) {
            if (stop) {
                break;
            }
            if (li < layerIdx) {
                continue;
            }
            double[] layer = layers.get(li);

            // Alteração aleatoria de todos os <TYPE>, <SLOPE>, <BIAS> e <WEIGHT>
            // Numero de neurons na camada
            int size = ((Double) layer[0]).intValue();

            // Numero de neurons na camada anterior (indica o numero de pesos para cada neuron)
            int inputs = ((Double) layer[1]).intValue();

            // <TYPE>
            int type = LAYER_FIELDS + IDX_N_TYPE;
            // <BIAS>
            int bias = LAYER_FIELDS + IDX_N_BIAS;
            // <WEIGHT>
            int weightStart = LAYER_FIELDS + IDX_N_WEIGHT;
            // Fator de incremento de fields do neuron
            int inc = inputs + NEURON_FIELDS;
            for (int ni = 0; ni < size; type += inc, bias += inc, weightStart += inc, ni++) {
                if (stop) {
                    break;
                }
                if (layerIdx == li && ni < neuronIdx) {
                    continue;
                }
                // Transfere os valores para a representaçao
                neuron.layer = li;
                neuron.layerSize = size;
                neuron.index = ni;
                neuron.type = Double.valueOf(layer[type]).intValue();
                neuron.bias = layer[bias];
                neuron.weights = new double[inputs];
                System.arraycopy(layer, weightStart, neuron.weights, 0, inputs);

                // Executa o callback
                stop = !callback.apply(neuron);

                // Transfere o valor da representação para o gene original
                if (Neuron.TYPE.getById(neuron.type) != null) {
                    layer[type] = (double) neuron.type;
                } else {
                    // Qualquer valor inválido é substituido pelo SIGMOID
                    layer[type] = Neuron.TYPE.SIGMOID.id;
                }
                layer[bias] = neuron.bias;

                // Se o novo neuron.weights for maior que o original, a mudança é descartada
                // Se o novo neuron.weights é menor que o original, o valor original é mantido
                for (int i = 0, l = neuron.weights.length, weight = weightStart; i < l && i < inputs; i++, weight++) {
                    layer[weight] = neuron.weights[i];
                }
            }
        }

        return layers.stream().flatMapToDouble(Arrays::stream).toArray();
    }

    /**
     * Obtém os valores mapeados das layers representados PELO DNA INFORMADO
     *
     * @param dna
     * @return
     */
    public static List<double[]> extractLayers(double[] dna) {
        final List<double[]> layers = new ArrayList<>();

        // ----------------
        // <SIZE>
        // <PREV>
        //     -- Neuron [0...SIZE]
        //     <TYPE>
        //     <BIAS>
        //     <WEIGHT> -- Peso para entrada [0...PREV]
        // ----------------
        for (int i = 0, l = dna.length; i < l;) {

            // Numero de neurons na camada
            int neurons = ((Double) dna[i]).intValue();

            // Numero de neurons na camada anterior (indica o numero de pesos para cada neuron)
            int inputs = ((Double) dna[i + 1]).intValue();

            // Quantidade de dados que pertencem a este layer
            int length = LAYER_FIELDS + neurons * NEURON_FIELDS + neurons * inputs;

            double[] layer = new double[length];
            System.arraycopy(dna, i, layer, 0, length);
            layers.add(layer);

            // Incrementa até o proximo layer
            i += length;
        }

        return layers;
    }

    /**
     * Obtém a quantidade de layers que um dna possui
     *
     * @param dna
     * @return
     */
    public static int[] countLayers(double[] dna) {
        final List<Integer> layers = new ArrayList<>();

        // ----------------
        // <SIZE>
        // <PREV>
        //     -- Neuron [0...SIZE]
        //     <TYPE>
        //     <BIAS>
        //     <WEIGHT> -- Peso para entrada [0...PREV]
        // ----------------
        for (int i = 0, l = dna.length; i < l;) {

            // Numero de neurons na camada
            int neurons = ((Double) dna[i]).intValue();

            // Numero de neurons na camada anterior (indica o numero de pesos para cada neuron)
            int inputs = ((Double) dna[i + 1]).intValue();

            // Quantidade de dados que pertencem a este layer
            int length = LAYER_FIELDS + neurons * NEURON_FIELDS + neurons * inputs;

            layers.add(neurons);

            // Incrementa até o proximo layer
            i += length;
        }

        return layers.stream().mapToInt(i -> i).toArray();
    }

    /**
     * Cria um cromossomo aleatório
     *
     * @param inputs Entradas da RN
     * @param outputs Saídas da RN
     * @return
     */
    public static Chromosome random(int inputs, int outputs) {
        // Número de layers
        int numLayers = Util.between(1, 3);

        // hidden + output
        int[] layers = new int[numLayers + 1];

        for (int i = 0; i < numLayers; i++) {
            // Numero de neuronios nos hidden layers está contido entre tamanho de entrada e saída+1 
            layers[i] = Util.between(Math.min(inputs, outputs), Math.max(inputs, outputs) + 1);
        }
        // Output layer
        layers[numLayers] = outputs;

        // Gera o DNA do cromossomo
        final List<Double> dna = new ArrayList<>();
        for (int i = 0, j = layers.length; i < j; i++) {
            int neurons = layers[i];
            int weights = i == 0 ? inputs : layers[i - 1];
            // <SIZE>
            dna.add(Double.valueOf(neurons));
            // <PREV>
            dna.add(Double.valueOf(weights));

            for (int k = 0; k < neurons; k++) {
                // <TYPE>
                dna.add(Neuron.TYPE.random());
                // <BIAS>
                dna.add(Util.between(BIAS_MIN, BIAS_MAX));
                for (int l = 0; l < weights; l++) {
                    // <WEIGHT>
                    dna.add(Util.between(WEIGHT_MIN, WEIGHT_MAX));
                }
            }
        }

        return new Chromosome(dna.stream().mapToDouble(i -> i).toArray());
    }

    /**
     * Representação OO de um Neuron
     */
    public static final class Neuron {

        public static enum TYPE {
            IDENTITY(Transfer.IDENTITY, 1.0),
            BENT_IDENTITY(Transfer.BENT_IDENTITY, 2.0),
            SIGMOID(Transfer.SIGMOID, 3.0),
            TANH(Transfer.TANH, 4.0),
            ARCTAN(Transfer.ARCTAN, 5.0),
            SOFTSIGN(Transfer.SOFTSIGN, 6.0),
            ISRU(Transfer.ISRU, 7.0),
            RELU(Transfer.RELU, 8.0),
            LRELU(Transfer.LRELU, 9.0),
            SOFTPLUS(Transfer.SOFTPLUS, 10.0),
            GAUSSIAN(Transfer.GAUSSIAN, 11.0);

            public final double id;
            public final Transfer transfer;

            private TYPE(Transfer transfer, double id) {
                this.id = id;
                this.transfer = transfer;
            }

            public static double random() {
                return Util.between(1.0, 11.0);
            }

            public static TYPE getById(double id) {
                for (TYPE type : TYPE.values()) {
                    if (type.id == id) {
                        return type;
                    }
                }
                return null;
            }

            public static TYPE getByFunction(Transfer transfer) {
                for (TYPE type : TYPE.values()) {
                    if (type.transfer == transfer) {
                        return type;
                    }
                }
                return null;
            }

        };
        /**
         * O número do layer a que este neuronio pertence
         */
        public int layer;

        /**
         * O tamanho (em neuronios) do layer a que este neuronio pertence
         */
        public int layerSize;

        /**
         * O índice deste neuronio no layer
         */
        public int index;

        /**
         * O tipo de função de transferencia do neuronio
         */
        public int type;

        /**
         * O bias do neuronio
         */
        public double bias;

        /**
         * Os pesos dos inputs deste neuronio
         */
        public double[] weights;

    }
}
