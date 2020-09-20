package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.Set;
import java.util.SortedSet;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.text.similarity.SimilarityScore;

import uk.co.terminological.datatypes.FluentMap;
import uk.co.terminological.datatypes.Triple;
import uk.co.terminological.datatypes.Tuple;

public class StringCrossMapper implements Serializable {

	Map<String,Document> sources = new LinkedHashMap<>();
	Map<String,Document> targets = new HashMap<>();
	Corpus sourceCorpus;
	Corpus targetCorpus;
	Normaliser normaliser;
	Tokeniser tokeniser;
	
	public static class DuplicateIdentityException extends RuntimeException {}
	
	public void addSource(String id, String source) {
		if (sources.containsKey(id)) throw new DuplicateIdentityException();
		this.sources.put(id, new Document(id, source, sourceCorpus));
	}
	
	public Corpus getSource() {return sourceCorpus;}
	public Corpus getTarget() {return targetCorpus;}
	
	public void addTarget(String id, String target) {
		if (targets.containsKey(id)) throw new DuplicateIdentityException();
		this.targets.put(id, new Document(id, target, targetCorpus));
	}
	
	public StringCrossMapper() {
		this(Collections.emptyList());
	}
	
	public StringCrossMapper(List<String> stopWords) {
		this(
			string -> string.replaceAll("[_,\\.]"," ").replaceAll("[^a-zA-Z0-9\\s]", "-").replaceAll("\\s+", " ").toLowerCase(),
			string -> Stream.of(string.split("\\s+")).filter(s -> !s.equals("-")),
			stopWords
		);
	}
	
	public StringCrossMapper(Normaliser normaliser, Tokeniser tokeniser, List<String> stopWords) {
		this.normaliser = normaliser;
		this.tokeniser = tokeniser;
		sourceCorpus = new Corpus(normaliser, tokeniser, stopWords);
		targetCorpus = new Corpus(normaliser, tokeniser, stopWords);
	}
	
	/**
	 * looks for documents which share terms with high tfidf scores 
	 * @param doc
	 * @return
	 */
	public Map<Document,Document> getBestMatches() {
		Map<Document,Document> match = new HashMap<>();
 		for (Entry<String,Document> source: sources.entrySet()) {
			getBestMatch(source.getValue()).ifPresent(
					doc2 -> match.put(source.getValue(), doc2)); 
		}
 		return match;
	}
	
	/*
	 * looks for documents which share terms with high tfidf score 
	 */
	private Optional<Document> getBestMatch(Document doc) {
		Iterator<Weighted<Term>> it = doc.termsByTfIdf().iterator();
		
		Collection<Document> matching = targetCorpus.getDocuments();
		
		double score = 0;
		while (it.hasNext() && matching.size() > 1) {
			Weighted<Term> next = it.next();
			Term nextTerm = next.getTarget();
			Optional<Term> outputTerm = targetCorpus.getMatchingTerm(nextTerm);
			if (outputTerm.isPresent()) {
				Set<Document> tmp = new HashSet<>(outputTerm.get().getDocumentsUsing());
				tmp.retainAll(matching); 
				if (tmp.size() > 0) {
					matching = tmp;
					score += next.getWeight();
				}
			}
		}
		
		if (matching.size() == 1) return matching.stream().findFirst();
		if (score == 0) return Optional.empty();
		Document bestMatch = null;
		
		Double bestScore = Double.NEGATIVE_INFINITY;
		for (Document match: matching) {
			
			Stream<Weighted<Term>> found = match.termsByTfIdf();
			Stream<Weighted<Term>> orig = doc.termsByTfIdf();
			
			Map<Term,Double> tmp = new HashMap<>();
			
			found.forEach(wt -> tmp.put(wt.getTarget(), -wt.getWeight()));
			orig.forEach(wt -> {
				tmp.merge(
						wt.getTarget(), 
						-wt.getWeight(), 
						(w1,w2) -> -1*(w1+w2) //if terms match then add tfidfs and change sign 
						);
			});
			
			//essentially I want to do a cross join on Terms here and combine weights
			
			Double tmpScore = tmp.values().stream().mapToDouble(d->d).sum();
			
			if (tmpScore > bestScore) {
				bestScore = tmpScore;
				bestMatch = match;
			}
		}
		
		return Optional.ofNullable(bestMatch);
		
	}
	
	/**
	 * A score between 0 and 1 where 1 is no difference and 0.5 is the median similarity score in the corpus
	 * @param minValue
	 * @return
	 */
	public List<Triple<Document,Document,Double>> getAllMatchesBySimilarity(Double minValue, 
			Function<Document,Stream<Weighted<Term>>> mapper, BiFunction<Stream<Weighted<Term>>,Stream<Weighted<Term>>,Double> algorithm) {
		//<Double> scores = new ArrayList<>(); 
		Map<Document,Set<Weighted<Document>>> match = new HashMap<>();
		Double max = Double.NEGATIVE_INFINITY;
		Double min = Double.POSITIVE_INFINITY;
		// int count = 0;
 		
 		for (Document doc: sourceCorpus.getDocuments()) {
 			Set<Document> targets = doc.getTerms().stream()
 					.flatMap(term -> targetCorpus.getMatchingTerm(term).stream())
 					.flatMap(term -> term.getDocumentsUsing().stream())
 					.collect(Collectors.toSet());
 			
 			SortedSet<Weighted<Document>> list = Weighted.descending();
 						
 			for (Document target: targets) {
 				Double distance = algorithm.apply(mapper.apply(doc),mapper.apply(target));
 				// scores.add(distance);
 				if (max < distance) max = distance; 
 				if (min > distance) min = distance;
 				// count += 1;
 				// add to or create the nested map
 				list.add(Weighted.create(target, distance));
 				
 			}
 			
 			match.put(doc, list);
 		}
 		
 		
 		final Double maxf = max;
 		List<Triple<Document,Document,Double>> out = new ArrayList<>();
 		match.forEach((doc1,v) -> v.forEach(w -> {
 			Double norm = 1-w.getWeight()/maxf;
 			if (norm>minValue) out.add(Triple.create(doc1,w.getTarget(),norm));
 		}));
 		
 		return out;
	}
	
	/**
	 * A score between 0 and 1 where 1 is no difference and 0.5 is the median similarity score in the corpus
	 * @param minValue
	 * @return
	 */
	public List<Tuple<Document,Document>> getTopMatchesBySimilarity(Integer matches, 
			Function<Document,Stream<Weighted<Term>>> mapper, BiFunction<Stream<Weighted<Term>>,Stream<Weighted<Term>>,Double> algorithm) {
		//<Double> scores = new ArrayList<>(); 
		List<Tuple<Document,Document>> out = new ArrayList<>();
		// int count = 0;
 		
 		for (Document doc: sourceCorpus.getDocuments()) {
 			Set<Document> targets = doc.getTerms().stream()
 					.flatMap(term -> targetCorpus.getMatchingTerm(term).stream())
 					.flatMap(term -> term.getDocumentsUsing().stream())
 					.collect(Collectors.toSet());
 			
 			List<Weighted<Document>> list = new ArrayList<>();
 						
 			for (Document target: targets) {
 				Double distance = algorithm.apply(mapper.apply(doc),mapper.apply(target));
 				// scores.add(distance);
 				// count += 1;
 				// add to or create the nested map
 				list.add(Weighted.create(target, distance));
 				Collections.sort(list);
 				list = list.subList(0, list.size() > matches ? matches : list.size());
 			}
 			
 			list.forEach(w -> out.add(Tuple.create(doc, w.getTarget())));
 		}
 		
 		return out;
	}	
	
	/**
	 * Computes the similarity for all combinations of source and target base on the raw text of the document, excluding normalisations
	 * @param minValue
	 * @param metric a class from org.apache.commons.text.similarity
	 * @return
	 */
	public <K extends Comparable<K>> Map<Document,Map<Document,K>> getAllMatchesByDistance(K maxValue, SimilarityScore<K> metric) {
		Map<Document,Map<Document,K>> match = new HashMap<>();
			
 		for (Document doc: sources.values()) {
 			Set<Document> targets = doc.getTerms().stream().flatMap(term -> targetCorpus.getMatchingTerm(term).stream()).flatMap(term -> term.getDocumentsUsing().stream()).collect(Collectors.toSet());
 			for (Document target: targets) {
 				K distance = getMatchByDistance(doc, target, metric); 
 				if (distance.compareTo(maxValue) < 0) {
 					Optional.ofNullable(match.get(doc)).ifPresentOrElse(
 	 						submap -> submap.put(target, distance),
 	 						() -> match.put(doc, FluentMap.with(target, distance)));
 				}
 			}
		}
 		
 		return match;
	}
	
	/*
	 * Takes an input document and calculates a similarity score for all the target documents based on the raw text of the document.
	 */
	private <K extends Comparable<K>> K getMatchByDistance(Document doc,Document target, SimilarityScore<K> similarity) {
		return similarity.apply(doc.getNormalised(), target.getNormalised());
	}
	
	public String summaryStats() {
		return new StringBuilder("Sources: ")
				.append("{"+sourceCorpus.summaryStats()+"}")
				.append(", Targets: ")
				.append("{"+targetCorpus.summaryStats()+"}").toString();
		
	}
	
	
}
