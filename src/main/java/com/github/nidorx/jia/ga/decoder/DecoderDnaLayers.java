package com.github.nidorx.jia.ga.decoder;

import com.github.nidorx.jia.ga.Chromosome;
import com.github.nidorx.jia.mlp.Layer;
import com.github.nidorx.jia.mlp.Neuron;
import java.util.ArrayList;
import java.util.List;

/**
 * Decodificador de DNA para lista de camdas de uma MLP
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class DecoderDnaLayers implements Decoder<Layer[], double[]> {

    private static class SingletonHelper {

        private static final DecoderDnaLayers INSTANCE = new DecoderDnaLayers();
    }

    public static DecoderDnaLayers getInstance() {
        return SingletonHelper.INSTANCE;
    }

    private DecoderDnaLayers() {
    }

    @Override
    public double[] encode(Layer[] layers) {
        final List<List<List<Double>>> encLayers = new ArrayList<>();

        for (Layer layer : layers) {
            List<List<Double>> encLayer = new ArrayList<>();

            layer.each((neuron, ni) -> {
                final List<Double> encNeuron = new ArrayList<>();

                Chromosome.Neuron.TYPE type = Chromosome.Neuron.TYPE.getByFunction(neuron.transfer);
                if (type == null) {
                    type = Chromosome.Neuron.TYPE.SIGMOID;
                }

                encNeuron.add(type.id);
                encNeuron.add(neuron.bias);

                for (double weight : neuron.weights) {
                    encNeuron.add(weight);
                }

                encLayer.add(encNeuron);
            });

            encLayers.add(encLayer);
        }

        return DecoderDnaList.getInstance().encode(encLayers);
    }

    @Override
    public Layer[] decode(double[] dna) {

        List<List<List<Double>>> encLayers = DecoderDnaList.getInstance().decode(dna);
        Layer[] layers = new Layer[encLayers.size()];

        for (int i = 0, j = encLayers.size(); i < j; i++) {
            List<List<Double>> encNeurons = encLayers.get(i);
            Neuron[] neurons = new Neuron[encNeurons.size()];

            for (int k = 0, l = encNeurons.size(); k < l; k++) {
                List<Double> encNeuron = encNeurons.get(k);

                Chromosome.Neuron.TYPE type = Chromosome.Neuron.TYPE.getById(encNeuron.get(Chromosome.IDX_N_TYPE));
                if (type == null) {
                    type = Chromosome.Neuron.TYPE.SIGMOID;
                }

                int prevSize = encNeuron.size() - Chromosome.NEURON_FIELDS;

                final Neuron neuron = new Neuron(prevSize, type.transfer);
                neuron.bias = encNeuron.get(Chromosome.IDX_N_BIAS);
                for (int m = 0, n = Chromosome.IDX_N_WEIGHT, o = encNeuron.size(); n < o; m++, n++) {
                    neuron.weights[m] = encNeuron.get(n);
                }

                neurons[k] = neuron;
            }
            layers[i] = new Layer(neurons);
        }

        return layers;
    }

}
