package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.SortedSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Topic implements Serializable {

	private int topicId;
	private SortedSet<Weighted<Term>> terms = Weighted.descending();
	private SortedSet<Weighted<Document>> documents = Weighted.descending();

	public Topic(int topic) {
		this.topicId = topic;
	}

	protected void addTerm(Weighted<Term> wt) {
		terms.add(wt);
	}
	
	public Stream<Weighted<Term>> streamTerms() {
		return terms.stream();
	}
	
	public Stream<Weighted<Document>> streamDocuments() {
		return documents.stream();
	}
	
	public int getTopicId() {return topicId;}

	protected void addDocument(Weighted<Document> create) {
		documents.add(create);
		create.getTarget().addTopic(Weighted.create(this, create.getWeight()));
	}
	
	public String toString() {
		return topicId+": "+terms.stream().limit(10).map(wt -> wt.getTarget().toString()).collect(Collectors.joining(" "));
	}
}
