package uk.co.terminological.nlptools;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.kennycason.kumo.CollisionMode;
import com.kennycason.kumo.WordCloud;
import com.kennycason.kumo.WordFrequency;
import com.kennycason.kumo.bg.CircleBackground;
import com.kennycason.kumo.bg.RectangleBackground;
import com.kennycason.kumo.font.KumoFont;
import com.kennycason.kumo.font.scale.SqrtFontScalar;
import com.kennycason.kumo.palette.ColorPalette;


public class WordCloudBuilder {

	private WordCloudBuilder() {}
	
	WordCloud wordCloud;
	List<WordFrequency> wordFrequencies = new ArrayList<>();
	//Path output;
	//ColorPalette pallette;
	Function<Corpus,Stream<Counted<Term>>> selector = c -> c.streamTerms().map(t -> Counted.create(t, t.countOccurrences()));
	Corpus corpus;
	int maxNumber;
	Dimension dimension;
	
	public static WordCloudBuilder from(Corpus corpus, int maxNumber, int x, int y) {
		WordCloudBuilder out = new WordCloudBuilder();
		out.corpus = corpus;
		out.maxNumber = maxNumber;
		out.dimension = new Dimension(x, y);
		out.wordCloud = new WordCloud(out.dimension, CollisionMode.PIXEL_PERFECT);
		out.wordCloud.setPadding(2);
		out.wordCloud.setKumoFont(new KumoFont(new Font("Lucida Sans", Font.PLAIN, 14)));
		out.wordCloud.setBackground(new CircleBackground(Math.min(x, y)/2));
		out.wordCloud.setFontScalar(new SqrtFontScalar(10, 50));
		
		out.withColourScheme(ColourScheme.Greys);
		return out;
	}
	
	public WordCloudBuilder withSelector(Function<Corpus,Stream<Counted<Term>>> mapper) {
		this.selector = mapper;
		return this;
	}
	
	/*public WordCloudBuilder withOutputPath(Path path) {
		this.output = path;
		return this;
	}*/
	
	public WordCloudBuilder withColourScheme(ColourScheme scheme) {
		List<Color> colors = scheme.values(8).stream().map(c -> c.toAwt()).collect(Collectors.toList());
		wordCloud.setColorPalette(new ColorPalette(colors));
		wordCloud.setBackgroundColor(scheme.background().toAwt()); //transparent black
		return this;
	}
	
	public WordCloudBuilder rectangular() {
		wordCloud.setBackground(new RectangleBackground(dimension));
		return this;
	}
	
	public WordCloudBuilder circular() {
		wordCloud.setBackground(new CircleBackground( (int) Math.floor(Math.min(dimension.getHeight(),dimension.getWidth())/2) ));
		return this;
	}
	
	public void execute(Path output) {
		selector.apply(corpus)
	       .sorted()
	       .filter(ct -> ct.getCount() > 0)
	       .limit(maxNumber)
	       .forEach((ti) -> {
	    	   wordFrequencies.add(new WordFrequency(ti.getTarget().getTag(), ti.getCount()));
		});
		wordCloud.build(wordFrequencies);
		wordCloud.writeToFile(output.toString());
	}
	
}
