package uk.co.terminological.nlptools.words;

import java.util.Set;

import com.koloboke.collect.set.hash.HashIntSets;

public class Ngram {

	int index;
	// int count=0;
	Set<Integer> descs = HashIntSets.newMutableSet(); 
	
	// Only construct via dictionary
	protected Ngram(int int1) {
		index = int1;
	}

	@Override
	public int hashCode() {
		return index;
	}


	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Ngram other = (Ngram) obj;
		if (index != other.index)
			return false;
		return true;
	}

	/*public void addOccurrence(Description desc) {
		count+=1;
		descs.add(desc.getId());
	}*/

	public int getId() {
		return index;
	}

}
