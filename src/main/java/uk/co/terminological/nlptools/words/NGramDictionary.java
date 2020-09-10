package uk.co.terminological.nlptools.words;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.koloboke.collect.map.hash.HashObjIntMaps;

import uk.co.terminological.nlptools.Normaliser;

public class NGramDictionary {

	Map<String,Integer> map = HashObjIntMaps.newMutableMap(100000);
	List<Ngram> strings = new ArrayList<>();
	Normaliser norm = Normaliser.DEFAULT;
	RareNgramIndex index = new RareNgramIndex();
	int nGramSize = 3;
	
	public Ngram lookup(int index) {
		return strings.get(index);
	}
	
	protected String ngramTerm(String term) {
		return new String(new char[nGramSize-1])+
				norm.apply(term)+
				new String(new char[nGramSize-1]);
	}
	
	public List<Integer> create(Description desc) {
		List<Integer> out = new ArrayList<>();
	
		String tmp = ngramTerm(desc.getTerm());
		for (int pos = 0; pos<(tmp.length()-nGramSize); pos++) {
			String characters = tmp.substring(pos, pos+nGramSize);
			Ngram ngram = null;
			if (map.containsKey(characters)) {
				int key = map.get(characters);
				ngram = strings.get(key);
				index.update(ngram, desc);
				out.add(key);
			} else {
				int key = strings.size();
				map.put(characters, key);
				ngram = new Ngram(key);
				index.update(ngram, desc);
				strings.add(ngram);
				out.add(key);
			}
		}
		
		
		
		return out;
	};
	
}
