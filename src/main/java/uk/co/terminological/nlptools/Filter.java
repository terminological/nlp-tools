package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

public interface Filter extends Predicate<String>,Serializable {

	public static Filter stopwords(List<String> stopwords, Normaliser normaliser, Tokeniser tokeniser) {
		Set<String> stopWordList = stopwords.stream()
				.map(normaliser)
				.flatMap(tokeniser)
				.collect(Collectors.toSet());
		return t -> stopWordList.contains(t);
	}
	
	public static Filter shorterThan(int size) {
		return t -> t.length() < size;
	}
	
	public static Filter number() {
		return t -> t.matches("[\\-+]?[0-9]*[\\.]*[0-9]+");
	}
	
}
