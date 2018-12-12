package com.github.nidorx.jia.ga.storage;

/**
 * Interface para persistencia de dados de execução do GA
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public interface Storage {

    /**
     * Faz o carregamento das informações de execução do GA
     *
     * @return
     * @throws java.lang.Exception
     */
    public Info load() throws Exception;

    /**
     * Permite salvar o GA. É acionado sempre que finaliza a execução de uma geração
     *
     * @param info
     * @throws java.lang.Exception
     */
    public void save(Info info) throws Exception;
}
