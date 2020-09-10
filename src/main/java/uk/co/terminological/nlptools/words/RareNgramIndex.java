package uk.co.terminological.nlptools.words;

import java.util.Arrays;

public class RareNgramIndex {

	Counter counts = new Counter(100000, 10000);
	
	public int getCounts(int ngramId) {
		return counts.value(ngramId);
	}
	
	public void update(Ngram ngram, Description desc) {
		int ngramId = ngram.getId();
		int prevfreq = counts.value(ngramId);
		int nextfreq = counts.increment(ngramId);
		//TODO:
		
		
		
		//ngram.descs
		
		// Mapping m = new Mapping(n, seq,	desc);
		/*freqs
			.entrySet()
			.stream()
			
			.forEach(kv -> {
				Ngram ngram = kv.getKey();
				int currentCount = kv.getValue();
				int sequence = ngrams.indexOf(o)
			});*/
		
	}
	
	
	/*private class Mapping {
		public Mapping(Ngram ng, int seq, Description desc) {
			this.nGramId = ng.getId();
			this.sequence = seq;
			this.descId = desc.getId();
		}
		int nGramId;
		int sequence;
		int descId;
	}*/
	
	private class Counter {
		
		private int[] values;
		private int growth;
		
		public Counter(int initialSize, int growth) {
			values = new int[initialSize];
			this.growth = growth;
		}
		
		public int value(int index) {
			return values[index];
		}
		
		public int increment(int index) {
			if (index > values.length) {
				values = Arrays.copyOf(values, index+growth); 
			}
			values[index] += 1;
			return values[index]; 
		}
		
	}

}
