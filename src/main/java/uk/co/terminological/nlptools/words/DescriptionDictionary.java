package uk.co.terminological.nlptools.words;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.koloboke.collect.map.hash.HashObjIntMaps;

public class DescriptionDictionary {

	Map<String,Integer> map = HashObjIntMaps.newMutableMap(100000);
	List<Description> strings = new ArrayList<>();
	
	public Description lookup(int index) {
		return strings.get(index);
	}
	
	public int put(Description desc) {
		Integer index = map.computeIfAbsent(desc.getTerm(), d -> {
			strings.add(desc);
			return strings.size()-1;
		});
		return index;
	}
	
	
	
}
