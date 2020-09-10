package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.SortedSet;
import java.util.stream.Stream;

import org.apache.commons.math3.distribution.ChiSquaredDistribution;

import cc.mallet.types.Token;

public class Term implements Serializable {

	int id;
	String tag;
	Corpus corpus;
	Set<Document> documentsUsing = new HashSet<>();
	Map<Term,Integer> cooccurrences = new HashMap<>();
	Set<TermInstance> instances = new HashSet<>();

	protected Term(String tag, Corpus map, int id) {
		this.tag = tag;
		this.corpus = map;
		this.id = id;
	}

	public int hashCode() {return id;}
	public boolean equals(Object o) {
		if (o instanceof Term) {
			return id == ((Term) o).id;
		} else return false;
	}
	public String toString() {return tag+" {idf:"+idf()+",entropy:"+shannonEntropy()+"}";} 



	/*
	 * Add a new document to this term - called when a term is found in a document during the Document constructor
	 * This is called after the document is set up but before the document is added to the corpus
	 */
	protected void add(Document doc) {
		documentsUsing.add(doc);
		for (Term other: doc.getTerms()) {
			if (!this.equals(other)) {
				cooccurrences.put(other, cooccurrences.getOrDefault(other, 0)+1);
			}
		}
	}

	protected void addInstance(TermInstance tmp2) {
		instances.add(tmp2);
	}

	public int countDocumentsWithTerm() {
		return documentsUsing.size();
	}

	public int countOccurrences() {
		return instances.size();
	}
	
	public double probabilityOccurrence() {
		return ((double) countOccurrences())/corpus.countCorpusTerms();
	}

	public Double shannonEntropy() {
		Double p = ((double) (countOccurrences()) / corpus.countCorpusTerms());
		return -p * Math.log(p) / Math.log(2D);
	}

	public Map<Term,Integer> cooccurrences() {
		return cooccurrences;
	}

	/**
	 * Pointwise mutual information can be normalized between [-1,+1] 
	 * resulting in -1 (in the limit) for never occurring together, 
	 * 0 for independence, and +1 for complete co-occurrence.[2]
	 * @return
	 */
	public Stream<Weighted<Term>> normalisedMutualInformation() {
		
		Integer total = corpus.countCorpusDocuments();
		SortedSet<Weighted<Term>> out = Weighted.descending();
		cooccurrences.forEach((k,cooccur) -> {
			Double mi = Calculation.npmi(cooccur, this.countDocumentsWithTerm(), k.countDocumentsWithTerm(), total);
			out.add(Weighted.create(k, mi));
		});
		return out.stream();
	}
	
	//TODO: this is mutual information for cooccurrence of terms within the document set we need to clarify this.
	public Stream<Weighted<Term>> mutualInformation() {
		
		Integer total = corpus.countCorpusDocuments();
		SortedSet<Weighted<Term>> out = Weighted.descending();
		cooccurrences.forEach((k,cooccur) -> {
			Double mi = Calculation.mi(cooccur, this.countDocumentsWithTerm(), k.countDocumentsWithTerm(), total);
			out.add(Weighted.create(k, mi));
		});
		return out.stream();
		
		//r.pmi = log( (toFloat(r.cooccurrences)*total) / (m.occurrences*n.occurrences) )
	    //r.probability = toFloat(r.cooccurrences)/total
	    //r.npmi = - log( (toFloat(r.cooccurrences)*total) / (m.occurrences*n.occurrences) ) / log ( toFloat(r.cooccurrences)/total )
	}
	

	public Stream<Weighted<Term>> cooccurenceProbablities() {
		Integer total = corpus.countCorpusDocuments();
		SortedSet<Weighted<Term>> out = Weighted.descending();
		cooccurrences.forEach((k,cooccur) -> {
			Double p = ((double) cooccur)/total;
			out.add(Weighted.create(k, p));
			
		});
		return out.stream();
	}

	
	/**
	 * The inverse document frequency is a measure of how much information the word provides, i.e., if it's common or rare across all documents. 
	 * <br/>
	 * It is the logarithmically scaled inverse fraction of the documents that contain the word 
	 * (obtained by dividing the total number of documents by the number of documents containing the term, and then taking the logarithm of that quotient):
	 * <br/>
	 * https://en.wikipedia.org/wiki/Tf%E2%80%93idf#Inverse_document_frequency
	 * <br/>
	 * This uses the 
	 * probabilistic inverse document frequency variant: log((N-nt)/nt)
	 * @return
	 */
	public Double idf() {
		return Math.log(((double) corpus.countCorpusDocuments())/this.countDocumentsWithTerm());
	}

	public Set<Document> getDocumentsUsing() {
		return documentsUsing;
	}

	public Map<Term, Integer> collocations(int spanLength) {
		Map<Term, Integer> out = new HashMap<>();
		this.instances.forEach(ti -> {
			ti.getNeighbours(spanLength).forEach(ti2 -> {
				Term t = ti2.getTerm();
				out.merge(t, 1, (oldV,newV) -> oldV+1);
			});
		});
		return out;
	}

	private static ChiSquaredDistribution csq = new ChiSquaredDistribution(1);
	
	/**
	 * Generates a set of p values for the 2 terms being correlated 
	 * @param spanLength
	 * @return
	 */
	public Stream<Weighted<Term>> chiSqCollocations(int spanLength) {
		SortedSet<Weighted<Term>> out = Weighted.descending();
		Integer N = this.corpus.countCorpusCollocations(spanLength);
		Map<Term,Integer> coll = collocations(spanLength);
		coll.forEach((term,count) -> {
			Integer o11 = count; //term1 and term2
			Integer o12 = this.countOccurrences() - count; //term1 present but not term2
			Integer o21 = term.countOccurrences() - count; //term2 present but not term1
			Integer o22 = N-this.countOccurrences()-term.countOccurrences()+count; // neither term present
			Double chiSq = ((double) N)*Math.pow((o11*o22-o12*o21),2)/((o11+o12)*(o11+o21)*(o12+o22)*(o21+o22));
			Double p = csq.cumulativeProbability(chiSq);
			out.add(Weighted.create(term, p));
		});
		return out.stream();
	}

	public Set<TermInstance> getInstances() {
		return instances;
	}
	
	protected Token asToken() {
		return new Token(tag);
	}

	public String getTag() {
		return tag;
	}

}