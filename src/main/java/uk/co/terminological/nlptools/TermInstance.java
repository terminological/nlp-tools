package uk.co.terminological.nlptools;

import java.io.Serializable;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

import cc.mallet.types.Token;
import uk.co.terminological.datatypes.FluentList;

public class TermInstance implements Serializable {

	private TermInstance previous = null;
	private Term current;
	private TermInstance next = null;
	
	public TermInstance(Term current) {
		this.current = current;
	}
	
	public Optional<TermInstance> getNext() {
		return Optional.ofNullable(next);
	}
	
	public Optional<TermInstance> getPrevious() {
		return Optional.ofNullable(previous);
	}
	
	public TermInstance(Term current, TermInstance previous) {
		this(current);
		this.previous = previous;
		this.getPrevious().ifPresent( p -> p.next = this );
	}
	
	public Term getTerm() {
		return current;
	}
	
	public boolean isFirst() {
		return !getPrevious().isPresent();
	}
	
	public boolean isLast() {
		return !getNext().isPresent();
	}
	
	public Set<TermInstance> getNeighbours(int n) {
		Set<TermInstance> out = new HashSet<>(getNext(n));
		out.addAll(getPrevious(n));
		return out;
	}
	
	public List<TermInstance> getNext(int n) {
		if (getNext().isPresent()) {
			if (n==1) {
				return FluentList.create(next);
			} else {
				List<TermInstance> tmp = getNext(n-1);
				if (tmp.size() == 0) {
					tmp.add(next); 
				} else {
					tmp.set(0, next);
				}
				return tmp;
			}
		} else {
			return FluentList.empty();
		}
	}
	
	public List<TermInstance> getPrevious(int n) {
		if (getPrevious().isPresent()) {
			if (n==1) {
				return FluentList.create(previous);
			} else {
				List<TermInstance> tmp = getPrevious(n-1);
				if (tmp.size() == 0) {
					tmp.add(previous); 
				} else {
					tmp.set(0, previous);
				}
				return tmp;
			}
		} else {
			return FluentList.empty();
		}
	}
	
	public String toString() {
		return getPrevious().map(ti -> ti.current.tag).orElse("X")+"-"+
				current.tag+"-"+
				getNext().map(ti -> ti.current.tag).orElse("X");
	}
	
	public Token asToken() {
		return this.current.asToken();
	}
}
