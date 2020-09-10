package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.SortedSet;
import java.util.TreeSet;

public class Weighted<T extends Serializable> implements Comparable<Weighted<T>>, Serializable {

	

	Double weight;
	T thing;
	
	public Double getWeight() {return weight;}
	public T getTarget() {return thing;}
	
	public Weighted(T thing, Double d) {
		this.thing = thing;
		this.weight = d;
	}
	
	public static <X extends Serializable> Weighted<X> create(X thing, Double d) {
		return new Weighted<X>(thing, d);
	}
	
	public String toString() {return thing.toString()+"(:"+weight.toString()+")";}
	
	public static <X extends Serializable> SortedSet<Weighted<X>> descending() {
		return new TreeSet<Weighted<X>>();
	}
	
	@Override
	public int compareTo(Weighted<T> o) {
		int comp = -getWeight().compareTo(o.getWeight());
		return comp == 0 ? hashCode()-o.hashCode() : comp;
	}
	
	public Counted<T> scale(int factor) {
		return Counted.create(this.getTarget(), (int) Math.ceil(this.getWeight()*factor));
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((thing == null) ? 0 : thing.hashCode());
		result = prime * result + ((weight == null) ? 0 : weight.hashCode());
		return result;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		@SuppressWarnings("unchecked")
		Weighted<T> other = (Weighted<T>) obj;
		if (thing == null) {
			if (other.thing != null)
				return false;
		} else if (!thing.equals(other.thing))
			return false;
		if (weight == null) {
			if (other.weight != null)
				return false;
		} else if (!weight.equals(other.weight))
			return false;
		return true;
	}
	
}
