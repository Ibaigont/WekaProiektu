package proiektu;

import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import java.io.File;

/**
 * Datu-multzoen karga eta kudeaketa egiten duen klasea.
 * CSV fitxategiak irakurtzen ditu, egitura egokitzen du (behar ez diren zutabeak kenduz)
 * eta Weka-ren {@link Instances} formatura pasatzen ditu.
 * * @version 1.0
 */
public class DatasetManager {

    /**
     * Garbitutako CSV fitxategi bat kargatzen du eta entrenamendurako prestatzen du.
     * TweetId eta TweetDate zutabeak kentzen ditu, eta 'Sentiment' zutabea klase-helburu gisa ezartzen du.
     *
     * @param cleanFile Aurretik garbitu den CSV fitxategia.
     * @return Entrenamendurako prest dauden datuak dituen {@link Instances} objektua.
     * @throws Exception Fitxategia irakurtzean edo atributuak kentzean erroreren bat gertatzen bada.
     */
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

    /**
     * Test-datuak kargatzen ditu eta entrenamendu-datuen egitura bera izan dezaten behartzen ditu.
     * Ezinbestekoa da eredua ondo aplikatzeko Test eta Train multzoek atributu berdinak izatea.
     *
     * @param cleanTestFile Aurretik garbitu den Test CSV fitxategia.
     * @param trainStructure Entrenamendu-datuen egitura (erreferentzia gisa erabiltzeko).
     * @return Test-datuak dituen eta entrenamenduaren egitura bera duen {@link Instances} objektua.
     * @throws Exception Fitxategia kargatzean edo instantziak berreraikitzean erroreren bat gertatzen bada.
     */
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