package proiektu;

import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;

public class DatasetManager {

    public static Instances loadTrainData(File cleanFile) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setStringAttributes("5"); // Tweetak String gisa irakurtzera behartzen du
        loader.setSource(cleanFile);
        Instances data = loader.getDataSet();

        // TweetId eta TweetDate kendu (3. eta 4. zutabeak)
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("3,4");
        removeFilter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, removeFilter);
        
        filteredData.setClassIndex(1);
        filteredData.deleteWithMissingClass();
        return filteredData;
    }

    public static Instances buildTestData(File cleanTestFile, Instances trainStructure) throws Exception {
        CSVLoader testLoader = new CSVLoader();
        testLoader.setStringAttributes("5");
        testLoader.setSource(cleanTestFile);
        Instances testDataRaw = testLoader.getDataSet();

        // Test instantziak training-aren egiturarekin berreraiki
        Instances testEgitura = new Instances(trainStructure, 0);
        for (int i = 0; i < testDataRaw.numInstances(); i++) {
            double[] vals = new double[trainStructure.numAttributes()];
            
            // 1. Topic (testDataRaw-en 0 zutabea da)
            String topicVal = testDataRaw.instance(i).stringValue(0);
            vals[0] = trainStructure.attribute(0).indexOfValue(topicVal);
            if (vals[0] < 0) vals[0] = Utils.missingValue();
            
            // 2. Sentiment
            String sentimentVal = testDataRaw.instance(i).stringValue(1);
            vals[1] = trainStructure.attribute(1).indexOfValue(sentimentVal);
            if (vals[1] < 0) vals[1] = Utils.missingValue();
            
            // 3. TweetText (testDataRaw-en 4 zutabea da, ez dugulako removeFilter erabili)
            String textVal = testDataRaw.instance(i).stringValue(4);
            vals[2] = testEgitura.attribute(2).addStringValue(textVal);
            
            testEgitura.add(new DenseInstance(1.0, vals));
        }
        testEgitura.setClassIndex(1);
        return testEgitura;
    }
}