package uk.co.terminological.nlptools.words;

import java.util.List;

public class Description {

	private int id;
	private String term;
	private Concept concept;
	private List<Integer> ngrams;
	
	protected Description(String term, Concept concept) {
		this.term = term;
		this.concept = concept;
		this.ngrams = concept.getConceptList().getNGramDict().create(this);
		this.id = this.concept.getConceptList().getDescDict().put(this);
	}

	public String getTerm() {
		return term;
	}

	public int getId() {
		return id;
	}
	
}
