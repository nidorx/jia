package com.github.nidorx.jia.util.function;

@FunctionalInterface
public interface TrheeFunctionThrowable<T, U, V, R> {

    R apply(T t, U u, V v) throws Exception;
}
