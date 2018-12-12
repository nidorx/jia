package com.github.nidorx.jia.util.function;

@FunctionalInterface
public interface BiFunctionThrowable<T, U, R> {

    R apply(T t, U u) throws Exception;
}
