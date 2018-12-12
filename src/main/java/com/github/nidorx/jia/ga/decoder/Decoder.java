package com.github.nidorx.jia.ga.decoder;

/**
 * Assinatura básica dos Decoders
 *
 * @author Alex Rodin <contato@alexrodin.info>
 * @param <D>
 * @param <E>
 */
public interface Decoder<D, E> {

    public E encode(D d);

    public D decode(E e);

}
