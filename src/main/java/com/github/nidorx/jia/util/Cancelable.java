package com.github.nidorx.jia.util;

/**
 * Permite cancelar uma ação
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
@FunctionalInterface
public interface Cancelable {

    public void cancel();
}
