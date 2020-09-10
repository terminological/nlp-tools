package uk.co.terminological.nlptools.words;

import java.util.ArrayList;

public class ConceptList extends ArrayList<Concept> {

	private DescriptionDictionary descDict;
	private NGramDictionary dict;
	
	protected DescriptionDictionary getDescDict() {
		return descDict;
	}
	
	protected NGramDictionary getNGramDict() {
		return dict;
	}
	
	public ConceptList() {
		this(new DescriptionDictionary(), new NGramDictionary() );
	}

	public ConceptList(DescriptionDictionary descriptionDictionary, NGramDictionary nGramDictionary) {
		this.descDict = descriptionDictionary;
		this.dict = nGramDictionary;
	}

	public Concept createConcept() {
		return new Concept(this);
	}
	
}
