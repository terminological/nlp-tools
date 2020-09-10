package uk.co.terminological.nlptools.words;

import java.util.ArrayList;
import java.util.List;

public class Concept {

	private ConceptList conceptList;
	private List<Description> descriptions = new ArrayList<>();
	
	protected Concept(ConceptList conceptList2) {
		conceptList = conceptList2;
	}
	
	protected ConceptList getConceptList() {
		return conceptList;
	}
	protected List<Description> getDescriptions() {
		return descriptions;
	}
	
	public Concept withDescription(String description) {
		Description tmp = new Description(description, this);
		descriptions.add(tmp);
		return this;
	}
}
