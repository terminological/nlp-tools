package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.topics.TopicModelDiagnostics;
import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;

// http://videolectures.net/mlss09uk_blei_tm/
// http://mimno.infosci.cornell.edu/topics.html
public class TopicModelResult implements Serializable {

	public TopicModelResult(ParallelTopicModel model, TopicModelBuilder builder) {
		super();
		this.model = model;
		this.builder = builder;

	}

	ParallelTopicModel model;
	TopicModelBuilder builder;
	
	public Alphabet getDataAlphabet() {
		return builder.instances.getDataAlphabet();
	};

	public void printTopics(int terms) {
		//PrintWriter pw = new PrintWriter(System.out);
		TopicModelDiagnostics diag = new TopicModelDiagnostics(model, terms);
		System.out.println(diag.toXML());
		//model.topicPhraseXMLReport(pw, 30);
		//pw.flush();
		//pw.close();
	}

	public int getTopicCount() {
		return builder.getTopics();
	}
	
	public Stream<Topic> getTopicsForDocuments() {
		
		if (!builder.getCorpus().hasTopics()) {
			ArrayList<TreeSet<IDSorter>> topicSortedWords = model.getSortedWords();
	        ArrayList<TreeSet<IDSorter>> topicSortedDocuments = model.getTopicDocuments(10.0);
			Alphabet alphabet = getDataAlphabet();
			
			int[] tokensPerTopic = model.tokensPerTopic;
			
	        // Print results for each topic
	        for (int topic = 0; topic < getTopicCount(); topic++) {
	            TreeSet<IDSorter> sortedWords = topicSortedWords.get(topic);
	            Topic top = builder.getCorpus().addTopic(topic);
	            
	            for (IDSorter info: sortedWords) {
	            	//Double weight = info.getWeight();
	            	Double probability = info.getWeight() / tokensPerTopic[topic];
	                String term = alphabet.lookupObject(info.getID()).toString();
	                Term t = builder.getCorpus().createTermFrom(term);
	                Weighted<Term> wt = Weighted.create(t, probability);
	                top.addTerm(wt);
	            }
	        
	            TreeSet<IDSorter> sortedDocuments = topicSortedDocuments.get(topic);
	            
	            for (IDSorter sorter: sortedDocuments) {
	                
	            	int doc = sorter.getID();
	                double proportion = sorter.getWeight();
	                String name = model.data.get(doc).instance.getName().toString();
	                
	                Optional<Document> optDoc = builder.getCorpus().getById(name);
	                optDoc.ifPresent(d -> {
	                	top.addDocument(Weighted.create(d, proportion));
	                });
	                
	            }
	        }
		}
		
        return builder.getCorpus().streamTopics();
		
		
	}

	//TODO: http://diging.github.io/tethne/tutorial.mallet.html#topic-modeling-in-mallet

	public Map<String, Double> getTopic(int topicNumber) {
		Map<String,Double> out = new HashMap<>();
		LabelSequence ls = model.data.get(topicNumber).topicSequence;
		for (int i=0; i<ls.getLength(); i++) {
			//ls.get
			//out.put(ls.getLabelAtPosition(i)., value)
		}
		return out;
	}

	public Corpus getCorpus() {
		return builder.getCorpus();
	}

	public SortedSet<Weighted<Integer>> predict(String string) {
		
		InstanceList testing = new InstanceList(builder.instances.getPipe());
		TokenSequence ts = new TokenSequence(getCorpus().process(string).map(s -> new Token(s)).collect(Collectors.toList()));
        testing.addThruPipe(new Instance(ts, null, "test instance", null));

        TopicInferencer inferencer = model.getInferencer();
        //MarginalProbEstimator estimator = model.getProbEstimator();
        double[] testProbabilities = inferencer.getSampledDistribution(testing.get(0), 10, 1, 5);
        
        //estimator.evaluateLeftToRight(testing, numParticles, usingResampling, docProbabilityStream);
        System.out.println("0\t" + testProbabilities[0]);
        SortedSet<Weighted<Integer>> out = Weighted.descending();
        for (int i =0 ; i<testProbabilities.length; i++) {
        	out.add(Weighted.create(i, testProbabilities[i]));
        }
		return out;
	}
}