package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.HashMap;
import java.util.stream.Collector;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import uk.co.terminological.datatypes.Tuple;



public class Similarity {

	/*
	 * Calculates a significance of similarity based on the tfidf
	 * For every term in the source document in order of descending tfidf
	 * Find documents containing that term in the target corpus
	 * Calculate the contribution that term will have on an overall score metric and multiply that to the current score for every matching document
	 * Move onto next term.
	 * 
	 */
	public static <X extends Serializable> Double getEuclideanDistance(Stream<Weighted<X>> source, Stream<Weighted<X>> target) {
		
		HashMap<X,Double> tmp = new HashMap<>();
		
		// copy first stream to vector
		source.forEach(kv -> tmp.put(kv.getTarget(), kv.getWeight()));
		// subtract second stream pairwise
		target.forEach(kv2 -> tmp.merge(kv2.getTarget(), kv2.getWeight(), (v1,v2) -> v1-v2));
		
		Double subSquares = tmp.entrySet().stream().collect(Collectors.summingDouble(kv -> kv.getValue()*kv.getValue()));
		return Math.sqrt(subSquares);
	}
	
	/*
	 * Calculates a cosine similarity based on a scored 
	 * For every term in the source document in order of descending tfidf
	 * Find documents containing that term in the target corpus
	 * Calculate the contribution that term will have on an overall score metric and multiply that to the current score for every matching document
	 * Move onto next term.
	 * 
	 */
	public static <X extends Serializable> Double getCosineDifference(Stream<Weighted<X>> source, Stream<Weighted<X>> target) {
		
		//List<Weighted<X>> tmpSource = new ArrayList<>();
		HashMap<X,Double> tmpSource = new HashMap<>();
		
		Double sourceLengthSqrd = source.collect(Collectors.summingDouble(v -> {
			tmpSource.put(v.getTarget(), v.getWeight());
			return v.getWeight()*v.getWeight();
		})); 
		
		Tuple<Double,Double> targetColl = target.collect(Collector.of(  
				  () -> Tuple.create(0D,0D),
				  (result, wX) -> {
					  Double lengthSqrd = result.getFirst();
					  Double dotProd = result.getSecond();
					  lengthSqrd += wX.getWeight()*wX.getWeight();
					  dotProd += wX.getWeight()*tmpSource.getOrDefault(wX.getTarget(),0D);
					  result.setFirst(lengthSqrd);
					  result.setSecond(dotProd);
				  },
				  (r1, r2) -> {
				    return Tuple.create(r1.getKey()+r2.getKey(), r1.getValue()+r2.getValue());
			}
		));
				
		Double targetLengthSqrd = targetColl.getFirst();
		Double dotProd = targetColl.getSecond();
		
		return 1 - dotProd / (Math.sqrt(sourceLengthSqrd*targetLengthSqrd));
	}
	
}
