package com.github.nidorx.jia.mlp;

import java.util.function.BiConsumer;

/**
 * Representação de uma camada de Neurons
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Layer {

    /**
     * Os neurons desta camadas
     */
    public final Neuron neurons[];

    /**
     * A dimensão desta camada
     */
    public final int size;

    public Layer(Neuron[] neurons) {
        this.size = neurons.length;
        this.neurons = neurons;
    }

    /**
     * Permite a criação de uma layer com a configuração informada
     *
     * @param size
     * @param prevLayerSize
     * @param type
     */
    public Layer(int size, int prevLayerSize, Transfer type) {
        this.size = size; // Bias
        this.neurons = new Neuron[size];
        for (int j = 0; j < size; j++) {
            neurons[j] = new Neuron(prevLayerSize, type);
        }
    }

    /**
     * Permite navegar em todos os neuronios desta camada
     *
     * @param fn
     */
    public void each(BiConsumer<Neuron, Integer> fn) {
        for (int i = 0; i < size; i++) {
            fn.accept(neurons[i], i);
        }
    }

    /**
     * Limpa os dados temporários de todos os neuronios desta camada
     */
    public void clear() {
        each((neuron, i) -> {
            neuron.clear();
        });
    }
}
