package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.function.Function;
import java.util.stream.Stream;

public interface Tokeniser extends Function<String,Stream<String>>,Serializable {
	
	public static Tokeniser DEFAULT = string -> Stream.of(string.split("\\s+")).filter(s -> !s.equals("-"));
	
}