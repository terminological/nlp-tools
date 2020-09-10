package uk.co.terminological.nlptools;

import java.io.IOException;
import java.io.Serializable;

public class SerialisationHelper implements Serializable {

	private String identifier;
	
	public SerialisationHelper(String id) {
		this.identifier=id;
	}

	//see https://bugs.openjdk.java.net/browse/JDK-8201131
	private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException { 
		in.defaultReadObject(); 
	}

	public String getIdentifier() {
		return identifier;
	}

	
}
