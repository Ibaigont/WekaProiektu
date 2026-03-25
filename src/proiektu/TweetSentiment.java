package proiektu;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;

import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.stemmers.SnowballStemmer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class TweetSentiment {

    public static void main(String[] args) {
        String csvPath     = "/home/ubuntu/Deskargak/Datuak Tweet sentiment-20260325/tweetSentiment.train.csv";
        String csvTestPath = "/home/ubuntu/Deskargak/Datuak Tweet sentiment-20260325/tweetSentiment.test_blind.csv";
        String outputPath  = "/home/ubuntu/git/WekaProiektu/Iragarpenak/iragarpenak.txt";

        try {
            // ── 1. TRAINING DATA kargatu ──────────────────────────────────────
            File cleanTrain = CSVPreprocessor.preprocessCSV(csvPath);
            CSVLoader loader = new CSVLoader();
            loader.setSource(cleanTrain);
            Instances data = loader.getDataSet();

            // ── 2. TweetId eta TweetDate kendu (3. eta 4. zutabeak) ───────────
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("3,4");
            removeFilter.setInputFormat(data);
            Instances dataFiltratuta = Filter.useFilter(data, removeFilter);
            // Orain: Topic(0) Sentiment(1) TweetText(2)
            dataFiltratuta.setClassIndex(1);

            // ── 3. StringToWordVector ─────────────────────────────────────────
            StringToWordVector STWV = new StringToWordVector();
            STWV.setAttributeIndices("last");
            STWV.setWordsToKeep(1000);
            STWV.setLowerCaseTokens(true);
            STWV.setTFTransform(true);
            STWV.setIDFTransform(true);
            SnowballStemmer stemmer = new SnowballStemmer();
            stemmer.setStemmer("english");
            STWV.setStemmer(stemmer);
            STWV.setInputFormat(dataFiltratuta);

            Instances dataBektorea = Filter.useFilter(dataFiltratuta, STWV);

            // ── 4. 10-fold cross-validation ───────────────────────────────────
            SMO smo = new SMO();
            smo.setC(1.0);
            Evaluation eval = new Evaluation(dataBektorea);
            eval.crossValidateModel(smo, dataBektorea, 10, new Random(1));
            System.out.println(eval.toSummaryString("Emaitz globalak", false));
            System.out.println(eval.toClassDetailsString("Klaseko xehetasunak"));
            System.out.println(eval.toMatrixString("Matrizea"));

            // ── 5. Modelo osoarekin entrenatu ─────────────────────────────────
            smo.buildClassifier(dataBektorea);

            // ── 6. TEST DATA kargatu ──────────────────────────────────────────
            File cleanTest = CSVPreprocessor.preprocessCSV(csvTestPath);
            CSVLoader testLoader = new CSVLoader();
            testLoader.setSource(cleanTest);
            Instances testData = testLoader.getDataSet();

            Instances testGarbitua = Filter.useFilter(testData, removeFilter);

            // Test instanceak training-aren egiturarekin berreraiki
            Instances testEgitura = new Instances(dataFiltratuta, 0);
            for (int i = 0; i < testGarbitua.numInstances(); i++) {
                double[] vals = new double[dataFiltratuta.numAttributes()];
                String topicVal = testGarbitua.instance(i).stringValue(0);
                vals[0] = dataFiltratuta.attribute(0).indexOfValue(topicVal);
                if (vals[0] < 0) vals[0] = Utils.missingValue();
                vals[1] = Utils.missingValue();
                vals[2] = testEgitura.attribute(2).addStringValue(
                              testGarbitua.instance(i).stringValue(2));
                testEgitura.add(new DenseInstance(1.0, vals));
            }
            testEgitura.setClassIndex(1);

            Instances testBektorea = Filter.useFilter(testEgitura, STWV);

            // ── 7. Iragarpenak idatzi ─────────────────────────────────────────
            try (FileWriter f = new FileWriter(outputPath)) {
                for (int i = 0; i < testBektorea.numInstances(); i++) {
                    double pred = smo.classifyInstance(testBektorea.instance(i));
                    String predLabel = dataBektorea.classAttribute().value((int) pred);
                    f.write(predLabel + "\n");
                }
            }
            System.out.println("Iragarpenak idatzita: " + outputPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}