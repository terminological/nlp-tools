package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.SortedSet;
import java.util.TreeSet;

public class Counted<T extends Serializable> implements Comparable<Counted<T>>, Serializable {

	

	Integer count;
	T thing;
	
	public Integer getCount() {return count;}
	public T getTarget() {return thing;}
	
	public Counted(T thing, Integer d) {
		this.thing = thing;
		this.count = d;
	}
	
	public static <X extends Serializable> Counted<X> create(X thing, Integer d) {
		return new Counted<X>(thing, d);
	}
	
	public String toString() {return thing.toString()+"(:"+count.toString()+")";}
	
	public static <X extends Serializable> SortedSet<Counted<X>> descending() {
		return new TreeSet<Counted<X>>();
	}
	
	@Override
	public int compareTo(Counted<T> o) {
		int comp = -getCount().compareTo(o.getCount());
		return comp == 0 ? hashCode()-o.hashCode() : comp;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((count == null) ? 0 : count.hashCode());
		result = prime * result + ((thing == null) ? 0 : thing.hashCode());
		return result;
	}
	
	public Weighted<T> asPercentage(int total) {
		return Weighted.create(thing, ((double) count)/total);
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
		Counted<T> other = (Counted<T>) obj;
		if (count == null) {
			if (other.count != null)
				return false;
		} else if (!count.equals(other.count))
			return false;
		if (thing == null) {
			if (other.thing != null)
				return false;
		} else if (!thing.equals(other.thing))
			return false;
		return true;
	}
	
	
	
}
