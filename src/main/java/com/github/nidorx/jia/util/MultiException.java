package com.github.nidorx.jia.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author Alex Rodin <contato@alexrodin.info>
 */
public class MultiException extends Exception {

    private static final long serialVersionUID = 1L;

    private final List<Throwable> errors;

    public MultiException(String message, List<Throwable> errors) {
        super(message);
        this.errors = new ArrayList<>(errors);
    }

    public List<Throwable> getFailures() {
        return Collections.unmodifiableList(errors);
    }

    @Override
    public String getMessage() {
        final StringBuilder sb = new StringBuilder(String.format("There were %d errors:", errors.size()));
        for (int i = 0, j = errors.size(); i < j; i++) {
            Throwable e = errors.get(i);
            sb.append(String.format("\n  % 2d. %s : %s", i + 1, e.getClass().getName(), e.getMessage()));
        }
        return sb.toString();
    }

}
