package com.github.nidorx.jia.ga;

import com.github.nidorx.jia.ga.decoder.DecoderDnaList;
import com.github.nidorx.jia.util.JiaUtils;
import java.util.ArrayList;
import java.util.List;

/**
 * Algoritmos de cruzamento entre dois cromossomos
 *
 @author Alex Rodin <contato@alexrodin.info>
 */
public class Crossover {

    private static final DecoderDnaList DECODER_DNA_LIST = DecoderDnaList.getInstance();

    /**
     * Executa um cruzamento aleatório, entre os algoritmos disponíveis
     *
     * @param dad
     * @param mom
     * @return
     */
    public static Chromosome random(Chromosome dad, Chromosome mom) {
        // int algorithm = Util.between(1, 4);
        // Somente singlePoint e twoPoints estão implementados
        int algorithm = JiaUtils.between(1, 2);
        switch (algorithm) {
            case 1:
                return singlePoint(dad, mom);
            case 2:
                return twoPoints(dad, mom);
            case 3:
                return uniform(dad, mom);
            default:
                return arithmetic(dad, mom);
        }
    }

    /**
     * Ponto de Cruzamento Único
     *
     * Um ponto de cruzamento é escolhido, a série binária desde o começo do cromossoma até o ponto de cruzamento é
     * copiada do primeiro pai e o resto copiado do outro pai
     * 
     *
     * @param dad
     * @param mom
     * @return
     */
    public static Chromosome singlePoint(Chromosome dad, Chromosome mom) {

        // Layers and neurons
        // [ 
        //   [ [<TYPE>, <BIAS>, <WEIGHT 0...n>], [<TYPE>, <BIAS>, <WEIGHT 0...n>] ],
        //   [ [<TYPE>, <BIAS>, <WEIGHT 0...n>], [<TYPE>, <BIAS>, <WEIGHT 0...n>] ],
        // ]
        List<List<List<Double>>> outData = new ArrayList<>();
        List<List<List<Double>>> dadData = DECODER_DNA_LIST.decode(dad.getDna());
        List<List<List<Double>>> momData = DECODER_DNA_LIST.decode(mom.getDna());

        int dadLayers = dadData.size();
        int momLayers = momData.size();

        // Escolhe o ponto de corte (um layer)
        final int ltLayers = Math.min(dadLayers, momLayers);
        int idxLayer = (int) JiaUtils.between(ltLayers * .30, ltLayers * 0.70);
        if (idxLayer == ltLayers) {
            idxLayer = ltLayers / 2;
        }

        // O menor layer será a primeira parte, o maior a segunda parte
        List<List<List<Double>>> ltData = dadLayers <= momLayers ? dadData : momData;
        List<List<List<Double>>> gtData = dadLayers > momLayers ? dadData : momData;

        for (int i = 0; i < idxLayer; i++) {
            outData.add(ltData.get(i));
        }

        for (int i = idxLayer, j = gtData.size(); i < j; i++) {
            outData.add(gtData.get(i));
        }

        // Gera o novo DNA a partir das informações dos neurons
        return new Chromosome(DECODER_DNA_LIST.encode(outData));
    }

    /**
     * Dois pontos de cruzamento]
     *
     * São definidos dois pontos de cruzamento, a série binária desde o início do cromossoma até o primeiro ponto de
     * cruzamento é copiada do primeiro pai, a parte do primeiro ponto de cruzamento até o segundo ponto é copiada do
     * outro pai e o resto do cromossoma é copiado do primeiro pai novamente
     *
     * @param dad
     * @param mom
     * @return
     */
    public static Chromosome twoPoints(Chromosome dad, Chromosome mom) {
        // Layers and neurons
        // [ 
        //   [ [<TYPE>, <BIAS>, <WEIGHT 0...n>], [<TYPE>, <BIAS>, <WEIGHT 0...n>] ],
        //   [ [<TYPE>, <BIAS>, <WEIGHT 0...n>], [<TYPE>, <BIAS>, <WEIGHT 0...n>] ],
        // ]
        List<List<List<Double>>> outData = new ArrayList<>();
        List<List<List<Double>>> dadData = DECODER_DNA_LIST.decode(dad.getDna());
        List<List<List<Double>>> momData = DECODER_DNA_LIST.decode(mom.getDna());

        List<List<List<Double>>> initData;
        List<List<List<Double>>> midData;
        if (JiaUtils.coin()) {
            // Pai no inicio e fim
            initData = dadData;
            midData = momData;
        } else {
            // Mae no inicio e fim
            initData = momData;
            midData = dadData;
        }

        int idxInitOne = Math.max((int) JiaUtils.between(initData.size() * .20, initData.size() * 0.50), 1);
        int idxInitTwo = Math.min((int) JiaUtils.between(initData.size() * .50, initData.size() * 0.80), initData.size() - 1);

        int idxMidStart = (int) JiaUtils.between(0, midData.size() * 0.40);
        int idxMidEnd = (int) JiaUtils.between(midData.size() * .60, midData.size());

        for (int i = 0; i < idxInitOne; i++) {
            outData.add(initData.get(i));
        }

        for (int i = idxMidStart; i < idxMidEnd; i++) {
            outData.add(midData.get(i));
        }

        for (int i = idxInitTwo, l = initData.size(); i < l; i++) {
            outData.add(initData.get(i));
        }

        // Gera o novo DNA a partir das informações dos neurons
        return new Chromosome(DECODER_DNA_LIST.encode(outData));
    }

    /**
     * Cruzamento Uniforme
     *
     * Os Neuron são copiados aleatóriamento do primeiro ou segundo pai
     *
     * @param dad
     * @param mom
     * @return
     */
    public static Chromosome uniform(Chromosome dad, Chromosome mom) {
        
        return null;
    }

    /**
     * Cruzamento Combinado
     *
     * Os cromossomos das camadas de mesmo índice são combinados
     *
     * @param dad
     * @param mom
     * @return
     */
    public static Chromosome merge(Chromosome dad, Chromosome mom) {
        return null;
    }

    /**
     * Cruzamento Aritmético
     *
     * É realizada uma operação aritmética em cada Neuron para obter a nova geração
     *
     * @param dad
     * @param mom
     * @return
     */
    public static Chromosome arithmetic(Chromosome dad, Chromosome mom) {
        return null;
    }
}
