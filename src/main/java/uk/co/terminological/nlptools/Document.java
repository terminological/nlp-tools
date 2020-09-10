package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.SortedSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.apache.commons.math3.util.CombinatoricsUtils;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;

import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.TokenSequence;

/**
 * A document is an identifiable string associated with a corpus
 */
public class Document extends SerialisationHelper implements Serializable {
	
	
	private String string;
	private String normalised;
	private List<TermInstance> termSequence = new ArrayList<>();
	private Map<Term,Integer> termCounts = new HashMap<>();
	private Corpus corpus;
	private Map<String,Object> metadata = new HashMap<>();
	private SortedSet<Weighted<Topic>> topics = Weighted.descending();
	
	protected Document(String id, String string, Corpus corpus) {
		super(id);
		this.corpus = corpus;
		this.string = string;
		//TODO: retain positional information
		//TODO: prenormalisation filtering.
		//TODO: sentence splitter
		//TODO: sentence normaliser not document normaliser
		Stream<String> tokens = corpus.process(string);
		TermInstance previous = null;
		Iterator<String> tokenIt = tokens.iterator(); 
		while (tokenIt.hasNext()) {
			String token = tokenIt.next();
			Term tmp = corpus.createTermFrom(token);
			TermInstance tmp2 = new TermInstance(tmp,previous);
			termSequence.add(tmp2);
			tmp.addInstance(tmp2);
			previous = tmp2;
			termCounts.put(tmp, termCounts.getOrDefault(tmp,0)+1);
		}
		// add document to the term - this generated some of the counts and co-occurences
		termCounts.keySet().forEach(tmp  -> tmp.add(this));
		this.corpus.addDocument(this);
	}
	
	// ============ BEAN METHODS =====================
	
	public String getString() {
		return string;
	}

	public String getNormalised() {
		return normalised;
	}

	public Set<Term> getTerms() {
		return termCounts.keySet();
	}

	public Corpus getCorpus() {
		return corpus;
	}
	
	public int hashCode() {
		return getIdentifier().hashCode();
	}
	
	public boolean equals(Object o) {
		if (o instanceof Document) return ((Document) o).getIdentifier().equals(this.getIdentifier());
		else return false;
	}
	
	public String toString() {
		return string+" ("+
				termSequence.stream().map(ti -> ti.getTerm())
					.map(t -> t.tag+"  {tfidf:"+tfIdf(t)+",entropy: "+t.shannonEntropy()+"}")
					.collect(Collectors.joining(","))+")";
	}
	
	// ============ SPECIFIC METHODS =====================
	
	public int countOccurencesInDocument(Term term) {
		return termCounts.get(term);
	}
	
	public int length() {
		return termSequence.size();
	}
	
	/**
	 * Calculate the tfidf fromt the frequency of terms in this document / the frequency of documents containing this term in the corpus
	 * <br/>
	 * In the case of the term frequency tf(t,d), the simplest choice is to use the raw count of a term in a document, i.e., the number of times that term t occurs in document d.<br\> 
	 * If we denote the raw count by ft,d, then the simplest tf scheme is tf(t,d) = ft,d. <br/>
	 * Other possibilities include:<br/>
	 * Boolean "frequencies": tf(t,d) = 1 if t occurs in d and 0 otherwise;<br/>
	 * term frequency adjusted for document length : ft,d ÷ (number of words in d)<br/>
	 * logarithmically scaled frequency: tf(t,d) = log (1 + ft,d)<br/>
	 * augmented frequency, to prevent a bias towards longer documents, <br/>
	 * e.g. raw frequency divided by the raw frequency of the most occurring term in the document:
	 * @param term
	 * @return
	 */
	public Double tfIdf(Term term) { 
		Double tf = ((double) countOccurencesInDocument(term));// / termsInDocumentTotal();
		if (term.tag.length() < 3) tf=tf/10;
		return tf*term.idf();
	}
	
	
	public Double shannonEntropy() {
		return termSequence.stream().map(ti -> ti.getTerm())
				.collect(Collectors.summingDouble(t -> t.shannonEntropy()));
	}
	
	/**
	 * returns the terms of the document in entropy descending order.
	 */
	public Stream<Term> entropyOrder() {
		List<Term> tmp = new ArrayList<>(termCounts.keySet());
		tmp.sort((t1,t2) -> t2.shannonEntropy().compareTo(t1.shannonEntropy()));
		return tmp.stream();
	}
	
	/**
	 * lists the terms in the document according to descending tfidf score.
	 * @return
	 */
	public Stream<Term> tfIdfOrder() {
		List<Term> orderedTerms = new ArrayList<>(termCounts.keySet());
		orderedTerms.sort((t1,t2) -> tfIdf(t2).compareTo(tfIdf(t1)));
		return orderedTerms.stream();
	}
	
	
	public Stream<Weighted<Term>> termsByTfIdf() {
		SortedSet<Weighted<Term>> tmp = Weighted.descending();
		termCounts.keySet().forEach(c -> tmp.add(Weighted.create(c, tfIdf(c))));
		return tmp.stream();
	}
	
	public Stream<Weighted<Term>> termsByEntropy() {
		SortedSet<Weighted<Term>> tmp = Weighted.descending();
		termCounts.keySet().forEach(c -> tmp.add(Weighted.create(c, c.shannonEntropy())));
		return tmp.stream();
	}
	
	/**
	 * possible collocations with max distance of spanLength between terms
	 * @param spanLength a number - typically larger than 1
	 * @return
	 */
	public int countCollocations(int spanLength) {
		int window = spanLength*2+1;
		if (window > length()) window = length();
		return length()*window-(window*(window+1))/2;
	}
	
	public int countCollocations(int spanLength, int sizeCollocations) {
		int window = spanLength*2+1;
		if (sizeCollocations > length() || sizeCollocations > window) return 0;
		if (sizeCollocations == 2) return countCollocations(spanLength);
		return (int) (countCollocations(spanLength)*CombinatoricsUtils.binomialCoefficient(window-2, sizeCollocations-2));
		
	}
	
	/**
	 * reconstructs a document from a list of terms and a separator
	 * @param terms
	 * @param sep
	 * @return A string of the document
	 */
	@Deprecated
	public static String termsToString(List<Term> terms, String sep) {
		return terms.stream().map(t -> t.tag).collect(Collectors.joining(sep));
	}
	
	protected Comparator<? super Term> descendingTfIdf() {return new Comparator<Term>() {
		@Override
		public int compare(Term t1, Term t2) {
			return Document.this.tfIdf(t2).compareTo(Document.this.tfIdf(t1));
		}
	};}

	protected TokenSequence asTokenSequence() {
		return new TokenSequence(this.termSequence.stream().map(ti -> ti.asToken()).collect(Collectors.toList()));
	}

	protected FeatureVector getFeatures() {
		Alphabet keys = corpus.getAlphabet();
		List<Integer> features = new ArrayList<>();
		List<Double> values = new ArrayList<>();
		this.metadata.forEach((k,v) -> {
			
			
			if (v != null) {
				
				Double value = null;
				if (v instanceof Boolean) value = ((boolean) v) ? 1D: 0D;
				if (v instanceof Double) value = ((double) v);
				if (v instanceof Integer) value = ((Integer) v).doubleValue();
				int feature = keys.lookupIndex(k, true);
				if (value != null) {
					features.add(feature);
					values.add(value);
				}
			}
			
			
		});
		return new FeatureVector(keys, Ints.toArray(features), Doubles.toArray(values));
		
	}

	public Document addMetadata(String string2, Object i) {
		this.metadata.put(string2, i);
		return this;
	}

	public void addTopic(Weighted<Topic> create) {
		this.topics.add(create);
	}
	
	public Stream<Weighted<Topic>> streamTerms() {
		return topics.stream();
	}

	public Optional<Object> getMetadata(String string2) {
		return Optional.ofNullable(this.metadata.get(string2));
	}

	public Map<Term,Integer> getTermCounts() {
		return termCounts;
	}

	
	
}