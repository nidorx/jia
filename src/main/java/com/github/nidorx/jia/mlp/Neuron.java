package com.github.nidorx.jia.mlp;

import java.util.Arrays;
import java.util.Random;

/**
 * Representação de um neuron
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class Neuron {

    /**
     * O tipo de função de transferencia do neuronio
     */
    public final Transfer transfer;

    /**
     * Pesos das conexões com a camada anterior
     */
    public double[] weights;

    /**
     * O bias do neuronio
     */
    public double bias;

    /**
     * O valor de saída do neuronio (após a propagação)
     */
    public double output;

    /**
     * O valor usado no método de ativação para obter a saída, usado no algoritmo de backpropagation como entrada da
     * função derivada
     */
    private double activation;

    /**
     * Mémória do delta usado na back propagation
     */
    public double delta;

    public Neuron(int prevLayerSize, Transfer type) {
        this(new Random().doubles(prevLayerSize, 0, 1).toArray(), Math.random(), type);
    }

    public Neuron(double[] weights, double bias, Transfer transfer) {
        this.weights = weights;
        this.bias = bias;
        this.transfer = transfer;
        this.delta = 0.0;
        this.output = 0.0;
    }

    /**
     * Calcula a ativação e a saída (transfer) do neuronio para as entradas informadas
     *
     * @param inputs
     * @return
     */
    public double activate(double[] inputs) {
        activation = 0d;
        try {
            for (int i = 0, l = weights.length; i < l; i++) {
                activation += weights[i] * inputs[i];
            }
        } catch (ArrayIndexOutOfBoundsException ex) {
            System.out.println("weights.length: " + weights.length + " - " + inputs.length);
            throw ex;
        }
        activation += bias; // Sum bias

        this.output = transfer.activation(activation);
        return this.output;
    }

    /**
     *
     * @param error cost derivative/ erro
     */
    public void gradient(double error) {
        this.delta = error * transfer.derivative(output, activation);
    }

    /**
     * Limpa os dados temporários do Neuron (delta e output)
     */
    public void clear() {
        this.delta = 0.0;
        this.output = 0.0;
    }

    @Override
    public String toString() {
        return "Neuron{"
                + "transfer=" + transfer
                + ", weights=" + Arrays.toString(weights)
                + ", bias=" + bias
                + ", output=" + output
                + ", delta=" + delta
                + '}';
    }

}
