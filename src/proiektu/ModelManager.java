package proiektu;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import java.util.Random;
import java.io.FileWriter;

public class ModelManager {
    private SMO bestModel;

    public void trainAndEvaluate(Instances trainData) throws Exception {
        System.out.println("\n========== 5 & 6. FASEAK: SVM-REN EBALUAZIOA ETA DOIKUNTZA ==========");
        double[] cValues = {0.1, 1.0, 5.0};
        double bestC = 1.0;
        double bestAccuracy = -1.0;
        
        for (double c : cValues) {
            SMO tempSmo = new SMO();
            tempSmo.setC(c);
            Evaluation evalTemp = new Evaluation(trainData);
            evalTemp.crossValidateModel(tempSmo, trainData, 10, new Random(1));
            
            double accuracy = evalTemp.pctCorrect();
            double fMeasure = evalTemp.weightedFMeasure();
            System.out.println("-> SMO (C=" + c + ") | Accuracy: " + String.format("%.2f%%", accuracy) + " | Weighted F-Measure: " + String.format("%.4f", fMeasure));
            
            if (accuracy > bestAccuracy) {
                bestAccuracy = accuracy;
                bestC = c;
            }
        }
        System.out.println("\n*** Aurkitutako C parametro onena: " + bestC + " ***");
        System.out.println("============================================================\n");

        bestModel = new SMO();
        bestModel.setC(bestC);
        Evaluation evalBest = new Evaluation(trainData);
        evalBest.crossValidateModel(bestModel, trainData, 10, new Random(1));
        System.out.println(evalBest.toSummaryString("Emaitz globalak (Eredu optimoa C=" + bestC + ")", false));
        System.out.println(evalBest.toClassDetailsString("Klaseko xehetasunak"));
        System.out.println(evalBest.toMatrixString("Matrizea"));

        // Eredu osoarekin entrenatu
        bestModel.buildClassifier(trainData);
    }

    public void testAndPredict(Instances testData, Instances trainData, String outputPath) throws Exception {
        Evaluation evalTest = new Evaluation(trainData);
        evalTest.evaluateModel(bestModel, testData);
        System.out.println("\n========== 8. FASEA: TEST DATUEN EBALUAZIOA (DEV CSV) ==========");
        System.out.println(evalTest.toSummaryString("Emaitz globalak (Test Datuetan)", false));
        System.out.println(evalTest.toClassDetailsString("Klaseko xehetasunak (Test Datuetan)"));
        System.out.println(evalTest.toMatrixString("Matrizea (Test Datuetan)"));
        System.out.println("=================================================================\n");

        try (FileWriter f = new FileWriter(outputPath)) {
            for (int i = 0; i < testData.numInstances(); i++) {
                double pred = bestModel.classifyInstance(testData.instance(i));
                String predLabel = trainData.classAttribute().value((int) pred);
                f.write(predLabel + "\n");
            }
        }
        System.out.println("Iragarpenak idatzita: " + outputPath);
    }
}