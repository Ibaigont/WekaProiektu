package proiektu;

import java.io.File;
import weka.filters.Filter;

import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TweetSentiment {
	public static void main(String[] args) {
		if (args.length!=1) {
			System.err.println("Erabilera: java -jar TweetSentiment.jar <csvPath>");
		}
		String csvPath = args[0];
		try {
			CSVLoader loader = new CSVLoader();
			loader.setSource(new File(csvPath));
			Instances data = loader.getDataSet();

			Remove removeFilter = new Remove();
			removeFilter.setAttributeIndices("3,4");
			removeFilter.setInputFormat(data);

			Instances dataFiltratuta = Filter.useFilter(data, removeFilter);

			dataFiltratuta.setClassIndex(1);

			StringToWordVector STWV = new StringToWordVector();
			STWV.setInputFormat(dataFiltratuta);
			STWV.setWordsToKeep(1000);
			STWV.setLowerCaseTokens(true);
			STWV.setTFTransform(true);
			STWV.setIDFTransform(true);

			SnowballStemmer stemmer = new SnowballStemmer();
			stemmer.setStemmer("english");
			STWV.setStemmer(stemmer);

			Instances dataBektorea = Filter.useFilter(dataFiltratuta, STWV);


		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
