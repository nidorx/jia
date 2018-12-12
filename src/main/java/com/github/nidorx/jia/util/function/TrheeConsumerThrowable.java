package com.github.nidorx.jia.util.function;

/**
 * Representa uma operação que recebe três argumentos de entrada e não possui retorno e pode lançar exceções.
 *
 * <p>
 * Essa é uma <strong>interface funcional</strong> cujo método funcional é {@link #accept(Object, Object, Object)}.
 *
 * @param <T> O tipo do primeiro argumento para a operação
 * @param <U> O tipo do segundo argumento para a operação
 * @param <V> O tipo do terceiro argumento para a operação
 *
 * @see java.util.function.Consumer
 */
@FunctionalInterface
public interface TrheeConsumerThrowable<T, U, V> {

    /**
     * Execute esta operação com os argumentos informados
     *
     * @param t O primeiro argumento de entrada
     * @param u O segundo argumento de entrada
     * @param v O terceiro argumento de entrada
     * @throws java.lang.Exception
     */
    void accept(T t, U u, V v) throws Exception;

}
