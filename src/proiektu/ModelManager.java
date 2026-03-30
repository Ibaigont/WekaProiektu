package proiektu;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.MultiFilter;
import java.util.Random;
import java.io.FileWriter;

public class ModelManager {
    // KONTUZ HEMEN: Orain ez da SMO bat gordetzen, FilteredClassifier bat baizik
    private FilteredClassifier bestModel;

    // Funtzioak datu gordinak eta iragazkien paketea jasotzen ditu
    public void trainAndEvaluate(Instances trainDataRaw, MultiFilter multiFilter) throws Exception {
        System.out.println("\n========== 5 & 6. FASEAK: FILTERED CLASSIFIER EBALUAZIOA ==========");
        double[] cValues = {0.1, 1.0, 5.0};
        double bestC = 1.0;
        double bestFMeasure = -1.0;
        
        for (double c : cValues) {
            // Oinarrizko eredua (SVM)
            SMO tempSmo = new SMO();
            tempSmo.setC(c);
            
            // FILTERED CLASSIFIER: Iragazkiak + Eredua elkartu
            FilteredClassifier fc = new FilteredClassifier();
            fc.setFilter(multiFilter);
            fc.setClassifier(tempSmo);
            
            // Ebaluazioa Cross-Validation bitartez (Datu gordinekin!)
            Evaluation evalTemp = new Evaluation(trainDataRaw);
            evalTemp.crossValidateModel(fc, trainDataRaw, 10, new Random(1));
            
            double accuracy = evalTemp.pctCorrect();
            double fMeasure = evalTemp.weightedFMeasure();
            System.out.println("-> FilteredClassifier SMO (C=" + c + ") | Accuracy: " + String.format("%.2f%%", accuracy) + " | Weighted F-Measure: " + String.format("%.4f", fMeasure));
            
            if (fMeasure > bestFMeasure) {
                bestFMeasure = fMeasure;
                bestC = c;
            }
        }
        System.out.println("\n*** Aurkitutako C parametro onena: " + bestC + " ***");
        System.out.println("============================================================\n");

        // Parametro optimoarekin eredu finala eratu
        SMO optimoSmo = new SMO();
        optimoSmo.setC(bestC);
        
        bestModel = new FilteredClassifier();
        bestModel.setFilter(multiFilter);
        bestModel.setClassifier(optimoSmo);
        
        // Eredu optimoaren ebaluazio osoa inprimatu
        Evaluation evalBest = new Evaluation(trainDataRaw);
        evalBest.crossValidateModel(bestModel, trainDataRaw, 10, new Random(1));
        System.out.println(evalBest.toSummaryString("Emaitz globalak (Eredu optimoa C=" + bestC + ")", false));
        System.out.println(evalBest.toClassDetailsString("Klaseko xehetasunak"));
        System.out.println(evalBest.toMatrixString("Matrizea"));

        // Eredu osoarekin entrenatu datu guztiekink (Gero test-erako erabiltzeko)
        bestModel.buildClassifier(trainDataRaw);
    }

    public void testAndPredict(Instances testDataRaw, Instances trainDataRaw, String outputPath) throws Exception {
        Evaluation evalTest = new Evaluation(trainDataRaw);
        // Test datu gordinak pasatzen ditugu! bestModel-ek bere barruan dituen iragazkiak aplikatuko dizkie bere kabuz
        evalTest.evaluateModel(bestModel, testDataRaw);
        
        System.out.println("\n========== 8. FASEA: TEST DATUEN EBALUAZIOA (DEV CSV) ==========");
        System.out.println(evalTest.toSummaryString("Emaitz globalak (Test Datuetan)", false));
        System.out.println(evalTest.toClassDetailsString("Klaseko xehetasunak (Test Datuetan)"));
        System.out.println(evalTest.toMatrixString("Matrizea (Test Datuetan)"));
        System.out.println("=================================================================\n");

        try (FileWriter f = new FileWriter(outputPath)) {
            for (int i = 0; i < testDataRaw.numInstances(); i++) {
                double pred = bestModel.classifyInstance(testDataRaw.instance(i));
                String predLabel = trainDataRaw.classAttribute().value((int) pred);
                f.write(predLabel + "\n");
            }
        }
        System.out.println("Iragarpenak idatzita: " + outputPath);
    }
}