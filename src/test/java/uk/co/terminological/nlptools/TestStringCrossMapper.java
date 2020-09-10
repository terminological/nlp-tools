package uk.co.terminological.nlptools;

import org.junit.Before;
import org.junit.Test;

import junit.framework.TestCase;

public class TestStringCrossMapper extends TestCase {

	static final String[] strings = {
			"one",
			"one two",
			"one two three",
			"two three four",
			"three four five",
			"four five",
			"five"
	};
	
	
	StringCrossMapper mapper;
	
	@Before
	public void setUp() throws Exception {
		super.setUp();
		this.mapper = new StringCrossMapper();
		for (int i=0; i<strings.length; i++) {
			this.mapper.addSource("S_"+i, strings[i]);
			this.mapper.addTarget("T_"+i, strings[i]);
		}
	}
	
	@Test
	public void testTfIDF() {
		mapper.getAllMatchesBySimilarity(0D, d -> d.termsByTfIdf(), Similarity::getCosineDifference).forEach(
				t -> System.out.println(t.getFirst().getIdentifier()+"\t"+t.getSecond().getIdentifier()+"\t"+t.getThird().toString())
			);
		
		System.out.println("=================");
		
		mapper.getAllMatchesBySimilarity(0D, d -> d.termsByEntropy(), Similarity::getEuclideanDistance).forEach(
				t -> System.out.println(t.getFirst().getIdentifier()+"\t"+t.getSecond().getIdentifier()+"\t"+t.getThird().toString())
			);
		
		System.out.println("=================");
		
		mapper.getSource().getTermsByTotalEntropy().forEach(term -> {
			System.out.println(term.toString());
		});
	}

	@Test
	public void testMutualInformation() {
		mapper.sourceCorpus.getMutualInformation().forEach(System.out::println);
	}
	
	@Test
	public void testCollocations() {
		mapper.sourceCorpus.getCollocations(2).forEach(System.out::println);
	}
	
}
