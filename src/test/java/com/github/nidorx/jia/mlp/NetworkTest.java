package com.github.nidorx.jia.mlp;

import com.github.nidorx.jia.util.Util;
import java.util.Arrays;
import org.junit.Assert;
import static org.junit.Assert.assertEquals;
import org.junit.Ignore;
import org.junit.Test;

/**
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class NetworkTest {

    public NetworkTest() {
    }

    @Test
    public void testForwardPropagation() throws Exception {
        // Baseado nos dados de 2.3. Forward Propagation,
        // disponível em https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

        Layer h = new Layer(new Neuron[]{
            new Neuron(new double[]{0.13436424411240122, 0.8474337369372327}, 0.763774618976614, Transfer.SIGMOID)
        });
        Layer o = new Layer(new Neuron[]{
            new Neuron(new double[]{0.2550690257394217}, 0.49543508709194095, Transfer.SIGMOID),
            new Neuron(new double[]{0.4494910647887381}, 0.651592972722763, Transfer.SIGMOID)
        });
        Network network = new Network(new Layer[]{h, o}, Arrays.asList(new String[]{"a", "b"}), Arrays.asList(new String[]{"a", "b"}));

        double[] output = network.input().set(new double[]{1.0, 0.0}).asArray();
        double[] expected = new double[]{0.6629970129852887, 0.7253160725279748};

        assertEquals(expected[0], output[0], 0.0);
        assertEquals(expected[1], output[1], 0.0);
    }

    @Test
    public void testBackPropagation() throws Exception {
        // Baseado nos dados de 3.2. Error Backpropagation
        // disponível em https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

        Neuron hn1 = new Neuron(new double[]{0.13436424411240122, 0.8474337369372327}, 0.763774618976614, Transfer.SIGMOID);
        hn1.output = 0.7105668883115941;
        Layer h = new Layer(new Neuron[]{hn1});

        Neuron on1 = new Neuron(new double[]{0.2550690257394217}, 0.49543508709194095, Transfer.SIGMOID);
        on1.output = 0.6213859615555266;

        Neuron on2 = new Neuron(new double[]{0.4494910647887381}, 0.651592972722763, Transfer.SIGMOID);
        on2.output = 0.6573693455986976;

        Layer o = new Layer(new Neuron[]{on1, on2});

        Network network = new Network(new Layer[]{h, o}, Arrays.asList(new String[]{"a", "b"}), Arrays.asList(new String[]{"a", "b"}));

        network.backPropagate(new double[]{0.0, 1.0});

        double[][][] expecteds = new double[][][]{
            // output, bias, delta,  weights[]
            new double[][]{
                new double[]{0.7105668883115941, 0.763774618976614, -0.0005348048046610517, 0.13436424411240122, 0.8474337369372327},},
            new double[][]{
                new double[]{0.6213859615555266, 0.49543508709194095, -0.14619064683582808, 0.2550690257394217},
                new double[]{0.6573693455986976, 0.651592972722763, 0.0771723774346327, 0.4494910647887381}
            }
        };
        network.each((layer, l) -> {
            layer.each((neuron, n) -> {
                double[] expected = expecteds[l][n];
                assertEquals(expected[1], neuron.bias, 0.0);
                assertEquals(expected[2], neuron.delta, 0.0);
                assertEquals(expected[0], neuron.output, 0.0);
                for (int i = 3, j = 0, k = neuron.weights.length; j < k; i++, j++) {
                    assertEquals(expected[i], neuron.weights[j], 0.0);
                }
            });
        });
    }

    /**
     * Teste do treinamento da MLP
     *
     * @throws java.lang.Exception
     */
    @Test
    @Ignore
    public void testTrain() throws Exception {
        Network network = Factory.build(new String[]{"a", "b"}, new int[]{4, 6, 2}, new String[]{"saida"}, Transfer.SIGMOID);

        network.setLearningRate(0.01);

        // Relaçao dataset e valor esperado
        final double values[][] = {
            {1.0, 1.0}, {2.0},
            {1.0, 2.0}, {3.0},
            {1.0, 3.0}, {4.0},
            {1.0, 4.0}, {5.0},
            {1.0, 5.0}, {6.0},
            {1.0, 7.0}, {8.0},
            {1.0, 8.0}, {9.0},
            {1.0, 9.0}, {10.0},
            {2.0, 2.0}, {4.0},
            {2.0, 3.0}, {5.0},
            {2.0, 4.0}, {6.0},
            {2.0, 5.0}, {7.0},
            {2.0, 6.0}, {8.0},
            {2.0, 7.0}, {9.0},
            {2.0, 8.0}, {10.0},
            {2.0, 9.0}, {11.0}
        };

        // Intervalo sendo trabalhado (entradas e saídas). Os valores são remapeados para o intervalo 0 e 1 para uso 
        // no MLP
        double min = 1;
        double max = 20;
        for (int i = 0, l = values.length; i < l; i++) {
            values[i] = Util.remap(values[i], min, max);
        }

        final double dataset[][] = new double[values.length / 2][values[0].length];
        final double expecteds[][] = new double[values.length / 2][values[1].length];

        for (int i = 0, d = 0, e = 0, j = values.length; i < j; i++) {
            if (i % 2 == 0) {
                dataset[d++] = values[i];
            } else {
                expecteds[e++] = values[i];
            }
        }

        // Executa o treinamento da rede neural
        final double maxErroAceitavel = 0.002;
        final int maximoEpocas = 100000000;
        network.train(dataset, expecteds, maxErroAceitavel, maximoEpocas);

        System.out.println(network.printNeurons());

        // Finalmente, verifica se a rede aprendeu a somar dois numeros
        validarSoma(network, new double[]{3.0, 3.0}, min, max);
        validarSoma(network, new double[]{4.0, 5.0}, min, max);
        validarSoma(network, new double[]{3.0, 9.0}, min, max);
        validarSoma(network, new double[]{7.0, 6.0}, min, max);
        validarSoma(network, new double[]{8.0, 7.0}, min, max);
    }

    private static void validarSoma(Network network, final double[] input, double min, double max) {
        final double[] copy = new double[]{input[0], input[1]};
        final double[] output = network.input().set(input, min, max).asArray(min, max);
        System.out.println(String.format("A doma de %.0f + %.0f = %.0f", copy[0], copy[1], output[0]));
        System.out.println(network.printGraph(min, max));
        Assert.assertEquals(Math.round(copy[0]) + Math.round(copy[1]), Math.round(output[0]));
    }

}
