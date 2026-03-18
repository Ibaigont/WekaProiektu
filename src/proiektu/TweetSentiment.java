package proiektu;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.output.prediction.CSV;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TweetSentiment {
	public static void main(String[] args) {
		//if (args.length!=3) {
		//	System.err.println("Erabilera: java -jar TweetSentiment.jar <csvPath> <csvTestPath> <outputPath>");
		//	return;
		//}
		String csvPath = "/home/ubuntu/Downloads/Datuak Tweet sentiment-20260318/tweetSentiment.train.csv";
		String csvTestPath = "/home/ubuntu/Downloads/Datuak Tweet sentiment-20260318/tweetSentiment.test_blind.csv";
		String outputPath = "iragarpenak.txt";
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

			SMO smo = new SMO();
			double paramC = 1.0;
			smo.setC(paramC);

			Evaluation eval = new Evaluation(dataBektorea);
			eval.crossValidateModel(smo, dataBektorea, 10, new Random(1));

			System.out.println(eval.toSummaryString("Emaitz globalak", false));
            System.out.println(eval.toClassDetailsString("Klaseko xehetasunak"));
            System.out.println(eval.toMatrixString("Matrizea"));
			smo.buildClassifier(dataBektorea);

			CSVLoader testLoader = new CSVLoader();
			testLoader.setSource(new File(csvTestPath));
			Instances testData = testLoader.getDataSet();

			Instances testGarbitua = Filter.useFilter(testData, removeFilter);
			testGarbitua.setClassIndex(1);
			Instances testBektorea = Filter.useFilter(testGarbitua, STWV);

			FileWriter f = new FileWriter(outputPath);

			for (int i = 0; i < testBektorea.numInstances(); i++) {
				double pred = smo.classifyInstance(testBektorea.instance(i));
				String predLabel = testBektorea.classAttribute().value((int) pred);
				f.write(predLabel + "\n");
			}
			f.close();





		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
