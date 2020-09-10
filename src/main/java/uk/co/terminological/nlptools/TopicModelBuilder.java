package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.ArrayList;

import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.InstanceList;

public class TopicModelBuilder implements Serializable {

	public static TopicModelBuilder create(Corpus corpus) {
		return new TopicModelBuilder(corpus);
	}

	private Corpus corpus;
	private ArrayList<Pipe> pipelist;
	InstanceList instances;
	private int topics = 5;

	private TopicModelBuilder(Corpus corpus) {

		this.setCorpus(corpus);
		this.pipelist = new ArrayList<>();
		this.pipelist.add( new TokenSequence2FeatureSequence() );
		this.instances = new InstanceList(new SerialPipes(pipelist));
	}

	public TopicModelBuilder withTopics(int topics) {
		this.setTopics(topics);
		return this;
	}

	public TopicModelResult execute(double alpha, double beta) {
		try {
			this.instances.addThruPipe(getCorpus().tokenSequenceIterator());
			ParallelTopicModel model = new ParallelTopicModel(getTopics(), alpha*getTopics(), beta);
			model.addInstances(instances);
			model.setNumThreads(2);
			model.setNumIterations(1000);
			model.estimate();
			return new TopicModelResult(model,this);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	@Deprecated
	public TopicModelResult executeDMR() {
		try {
		this.instances.addThruPipe(getCorpus().tokenSequenceIterator());
		ParallelTopicModel model = new ParallelTopicModel(getTopics());
		model.addInstances(instances);
		model.setNumThreads(2);
		model.setNumIterations(1000);
		model.estimate();
		return new TopicModelResult(model,this);
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException(e);
		}
	}

	public int getTopics() {
		return topics;
	}

	public void setTopics(int topics) {
		this.topics = topics;
	}

	public Corpus getCorpus() {
		return corpus;
	}

	public void setCorpus(Corpus corpus) {
		this.corpus = corpus;
	}


}
