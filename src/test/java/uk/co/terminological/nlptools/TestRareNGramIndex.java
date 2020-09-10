package uk.co.terminological.nlptools;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import uk.co.terminological.nlptools.words.Concept;
import uk.co.terminological.nlptools.words.ConceptList;

public class TestRareNGramIndex {

	@Before
	public void setUp() throws Exception {
	}

	@After
	public void tearDown() throws Exception {
	}

	@Test
	public void test() {
		ConceptList cl = new ConceptList();
		Concept con = cl.createConcept().withDescription("an example description");
		con.hashCode();
	}

}
