package com.github.nidorx.jia.mlp;

import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Funções de ativação dos neuronios
 *
 * @see https://sourceforge.net/p/neuroph/code/HEAD/tree/trunk/neuroph-2.8/Core/src/main/java/org/neuroph/core/transfer/
 */
public class Transfer {

    /**
     * Função identidade (linear)
     *
     * A saída de uma função de identidade é igual à sua entrada
     *
     * @see https://en.wikipedia.org/wiki/Identity_function
     */
    public static final Transfer IDENTITY = new Transfer(
            "IDENTITY",
            // f(x) = x
            (x) -> x,
            // f(x)' = 1 
            (output, x) -> 1.0
    );

    /**
     * Sigmoid/Logistic Function
     *
     * A função Sigmoid converte os valores de entrada (que pode variar entre -∞ a +∞) e espreme todos os valores entre
     * o intervalo de 0 a 1.
     *
     * @see https://en.wikipedia.org/wiki/Logistic_function
     */
    public static final Transfer SIGMOID = new Transfer(
            "SIGMOID",
            // f(x) = 1 ÷ (1 + e^-x)
            (x) -> 1.0 / (1.0 + Math.exp(-x)),
            // f(x)' = f(x) * (1 - f(x))
            (output, x) -> output * (1.0 - output)
    );

    /**
     * Hyperbolic tangent
     *
     * @see https://en.wikipedia.org/wiki/Hyperbolic_function#Hyperbolic_tangent
     */
    public static final Transfer TANH = new Transfer(
            "TANH",
            // f(x) = tanh(x) = (2 ÷ (1 + e^(-2x)))-1
            (x) -> Math.tanh(x),
            // f(x)' = 1 - f(x)^2
            (output, x) -> 1 - Math.pow(output, 2)
    );

    /**
     * ArcTangent
     *
     * @see https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
     */
    public static final Transfer ARCTAN = new Transfer(
            "ARCTAN",
            // f(x) = tan^-1(x)
            (x) -> Math.atan(x),
            // f(x)' = 1 / (x^2 + 1)
            (output, x) -> 1 / (Math.pow(x, 2) + 1)
    );

    /**
     * Softsign
     */
    public static final Transfer SOFTSIGN = new Transfer(
            "SOFTSIGN",
            // f(x) = x ÷( 1 + |x|)
            (x) -> x / (1 + Math.abs(x)),
            // f(x)' = 1 ÷ (1 + |x|)^2
            (output, x) -> 1 / Math.pow((1 + Math.abs(output)), 2)
    );

    /**
     * Inverse square root unit
     *
     * The ISRLU hyperparameter α controls the value to which an ISRLU saturates for negative inputs
     *
     * @see https://openreview.net/pdf?id=HkMCybx0-
     */
    public static final Transfer ISRU = new Transfer(
            "ISRU",
            // α = slope = 1
            // f(x) = x(1 ÷ Sqrt(1 + (αx)^2))
            (x) -> x * fastInvSqrt(1 + Math.pow(x, 2)),
            // f(x)' = (1 ÷ Sqrt(1 + αx^2))^3
            // f(x)' = 1 ÷ (x^2+1)^(3÷2)
            (output, x) -> Math.pow(fastInvSqrt(1 + Math.pow(x, 2)), 3)
    );

    /**
     * Inverse square root linear unit
     *
     * @see https://openreview.net/pdf?id=HkMCybx0-
     */
    public static final Transfer ISRLU = new Transfer(
            "ISRLU",
            // a = slope = 3
            // f(x) = x if x ≥ 0; ISRU(x) if x < 0
            (x) -> x >= 0 ? x : ISRU.activation(x),
            // f(x)' = 1 if x ≥ 0; ISRU(x)' if x < 0
            (output, x) -> x >= 0 ? 1 : ISRU.derivative(output, x)
    );

    /**
     * Exponential linear unit
     */
    public static final Transfer ELU = new Transfer(
            "ELU",
            // f(x) = α(e^x - 1) if x < 0; x if x ≥ 0
            // α = 1
            (x) -> x < 0 ? (Math.pow(Math.E, x) - 1) : x,
            // f(x)' = f(α,x) + α if x < 0; 1 if x ≥ 0
            (output, x) -> x < 0 ? output + 1 : 1
    );

    /**
     * Rectified linear unit
     *
     * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
     */
    public static final Transfer RELU = new Transfer(
            "RELU",
            // f(x) = 0 if x < 0; x if x ≥ 0
            (x) -> x < 0 ? 0 : x,
            // f(x)' = 0 if x < 0; 1 if x ≥ 0
            (output, x) -> x < 0 ? 0 : 1.0
    );

    /**
     * Leaky rectified linear unit
     *
     * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
     */
    public static final Transfer LRELU = new Transfer(
            "LRELU",
            // f(x) = αx if x < 0; x if x ≥ 0
            // α = 0.2
            (x) -> x < 0 ? 0.2 * x : x,
            // f(x)' = 0.01 if x < 0; 1 if x ≥ 0
            (output, x) -> x < 0 ? 0.01 : 1.0
    );

    /**
     * SoftPlus
     *
     * @see https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
     */
    public static final Transfer SOFTPLUS = new Transfer(
            "SOFTPLUS",
            // f(x) = ln(1 + e^x)
            (x) -> Math.log1p(Math.exp(x)),
            // f(x)' = 1 ÷ (1 + e^-x)
            (output, x) -> 1.0 / (1 + Math.exp(-x))
    );

    /**
     * Bent identity
     *
     * @see https://en.wikipedia.org/wiki/Activation_function
     */
    public static final Transfer BENT_IDENTITY = new Transfer(
            "BENT_IDENTITY",
            // f(x) = ((Sqrt(x^2 + 1) - 1) ÷ 2) + x
            (x) -> ((Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2) + x,
            // f(x)' = (x ÷ (2 * Sqrt(x^2 + 1))) + 1
            (output, x) -> (x / (2 * Math.sqrt(Math.pow(x, 2) + 1))) + 1
    );

    /**
     * Gaussian function
     *
     * @see https://en.wikipedia.org/wiki/Gaussian_integral
     * @see https://en.wikipedia.org/wiki/Activation_function
     */
    public static final Transfer GAUSSIAN = new Transfer(
            "GAUSSIAN",
            // f(x) = e^(-(x^2))
            (x) -> Math.pow(Math.E, -Math.pow(x, 2)),
            // f(x)' = -2xe^(-(x^2))
            (output, x) -> -2 * x * output
    );

    public final String name;
    private final UnaryOperator<Double> activation;
    private final BinaryOperator<Double> derivative;

    /**
     *
     * @param name
     * @param activation
     * @param derivative Derivativa da função já esperando o valor passado como sendo o resultado da função de ativação
     * (evita recalcular em alguns algoritmos)
     */
    private Transfer(String name, UnaryOperator<Double> activation, BinaryOperator<Double> derivative) {
        this.name = name;
        this.activation = activation;
        this.derivative = derivative;
    }

    /**
     * @param activation
     * @return
     */
    public double activation(double activation) {
        return this.activation.apply(activation);
    }

    /**
     *
     * @param output Saída de {@link Transfer#activation(double)}
     * @param activation Entrada de {@link Transfer#activation(double)}, pode ser usado em algum algoritmo
     * @return
     */
    public double derivative(double output, double activation) {
        return this.derivative.apply(output, activation);
    }

    @Override
    public String toString() {
        return name;
    }

    /**
     *
     * @param x
     * @return
     *
     * @see https://en.wikipedia.org/wiki/Fast_inverse_square_root
     */
    private static double fastInvSqrt(double x) {
        double xhalf = 0.5d * x;
        long i = Double.doubleToLongBits(x);
        i = 0x5fe6ec85e7de30daL - (i >> 1);
        x = Double.longBitsToDouble(i);
        x *= (1.5d - xhalf * x * x);
        return x;
    }

}
