package uk.co.terminological.nlptools;

import org.junit.Test;

public class TestCalculation {

	@Test
	public void testPMI() {
		double pmi = Calculation.pmi(1159, 1938, 1311, 50000952);
		System.out.println(pmi);
		assert(Math.abs(pmi-10.0349081703D) < 0.0001 );
	}

	@Test
	public void testMI() {
		double mi = Calculation.mi(0.05, 0.15, 0.7);
		System.out.println(mi);
		assert(Math.abs(mi - Math.log(Math.pow(2.0, 0.2141709D))) < 0.0001 );
	}
	
}
