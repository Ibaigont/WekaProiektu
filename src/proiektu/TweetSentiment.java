package proiektu;

import java.io.File;
import java.io.FileWriter;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Random;

import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.instance.ClassBalancer;

public class TweetSentiment {

    public static void main(String[] args) {
        // --- Experimentatzeko balioak ---
        boolean stemmerErabili = true;
        boolean stopWordsErabili = true;
        boolean bigramakErabili = true;
        int hiztegiTamaina = 3000;
        boolean atributuakOptimizatu = false; // Desactivado para soportar pesos de ClassBalancer
        int azkenAtributuKopurua = 1500;
        boolean klaseakOrekatu = true;
        // --------------------------------------------------

        String csvPath = "data/tweetSentiment.train.csv";
        String csvTestPath = "data/tweetSentiment.dev.csv";
        String outputPath = "Iragarpenak/iragarpenak.txt";
        String logPath = "Iragarpenak/informe_resultados.txt";

        // --- 9. FASEA: KOMANDO BIDEZKO EXEKUZIOA ---
        if (args.length >= 4) {
            csvPath = args[0];
            csvTestPath = args[1];
            outputPath = args[2];
            logPath = args[3];
        } else if (args.length >= 3) {
            csvPath = args[0];
            csvTestPath = args[1];
            outputPath = args[2];
        }

        PrintStream jatorrizkoKontsola = System.out;
        try {
            System.setOut(new PrintStream(new FileOutputStream(logPath)));
            jatorrizkoKontsola.println("\n[+] Weka pipeline-a exekutatzen...");
            jatorrizkoKontsola.println(
                    "[+] Mesedez, itxaron segundo batzuk STWV-k hiztegia sortu eta SVM Cross-Validation bidez optimizatzen den bitartean...");
            jatorrizkoKontsola.println(
                    "[+] KONTUZ: Kontsola isilik egongo da, analisi GUZTIAK zuzenean gordetzen ari direlako hemen: "
                            + logPath + "\n");

            // ── 1. TRAINING DATA kargatu ──────────────────────────────────────
            File cleanTrain = CSVPreprocessor.preprocessCSV(csvPath, true);
            CSVLoader loader = new CSVLoader();
            loader.setStringAttributes("5"); // OSO GARRANTZITSUA: Tweetak String gisa irakurtzera behartzen du, ez
                                             // Nominal gisa
            loader.setSource(cleanTrain);
            Instances data = loader.getDataSet();

            // ── 2. TweetId eta TweetDate kendu (3. eta 4. zutabeak) ───────────
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("3,4");
            removeFilter.setInputFormat(data);
            Instances dataFiltratuta = Filter.useFilter(data, removeFilter);
            // Orain: Topic(0) Sentiment(1) TweetText(2)
            dataFiltratuta.setClassIndex(1);
            dataFiltratuta.deleteWithMissingClass(); // Galdutako klasea duten instantziak garbitu InfoGain-en
                                                     // kraskadurak saihesteko

            if (klaseakOrekatu) {
                // ClassBalancer emplea pesos matemáticos, por lo que no duplica datos ni causa overfitting
                ClassBalancer balancer = new ClassBalancer();
                balancer.setInputFormat(dataFiltratuta);
                dataFiltratuta = Filter.useFilter(dataFiltratuta, balancer);
            }

            // --- 1. FASEA: Datu-sortaren analisia ---
            System.out.println("\n========== 1. FASEA: DATU-SORTAREN ANALISIA ==========");
            System.out.println("Instantzia kopurua (Training): " + dataFiltratuta.numInstances());
            System.out.println("Atributu kopurua: " + dataFiltratuta.numAttributes());
            System.out.println("Atributuak:");
            for (int i = 0; i < dataFiltratuta.numAttributes(); i++) {
                System.out.println("  " + i + ". " + dataFiltratuta.attribute(i).name() +
                        " (Mota: " + weka.core.Attribute.typeToString(dataFiltratuta.attribute(i)) + ")");
            }
            System.out.println("\nKlaseen banaketa (Sentiment):");
            int[] classCounts = dataFiltratuta.attributeStats(dataFiltratuta.classIndex()).nominalCounts;
            for (int i = 0; i < dataFiltratuta.numClasses(); i++) {
                System.out.println(
                        "  Klasea " + dataFiltratuta.classAttribute().value(i) + ": " + classCounts[i] + " instantzia");
            }
            System.out.println("==================================================\n");

            // ── 3. StringToWordVector ─────────────────────────────────────────
            StringToWordVector STWV = new StringToWordVector();
            STWV.setAttributeIndices("last");
            STWV.setWordsToKeep(hiztegiTamaina);
            STWV.setLowerCaseTokens(true);
            STWV.setTFTransform(true);
            STWV.setIDFTransform(true);

            if (bigramakErabili) {
                weka.core.tokenizers.NGramTokenizer tokenizer = new weka.core.tokenizers.NGramTokenizer();
                tokenizer.setNGramMinSize(1);
                tokenizer.setNGramMaxSize(2);
                STWV.setTokenizer(tokenizer);
            }

            if (stemmerErabili) {
                weka.core.stemmers.IteratedLovinsStemmer stemmer = new weka.core.stemmers.IteratedLovinsStemmer();
                STWV.setStemmer(stemmer);
            }

            if (stopWordsErabili) {
                weka.core.stopwords.Rainbow stopwords = new weka.core.stopwords.Rainbow();
                STWV.setStopwordsHandler(stopwords);
            }

            STWV.setInputFormat(dataFiltratuta);

            Instances dataBektorea = Filter.useFilter(dataFiltratuta, STWV);

            // ── 4. FASEA: Atributuen hautaketa ────────────────────────────────
            weka.filters.supervised.attribute.AttributeSelection attrSel = new weka.filters.supervised.attribute.AttributeSelection();
            if (atributuakOptimizatu) {
                weka.attributeSelection.InfoGainAttributeEval evalInfo = new weka.attributeSelection.InfoGainAttributeEval();
                weka.attributeSelection.Ranker ranker = new weka.attributeSelection.Ranker();
                ranker.setNumToSelect(azkenAtributuKopurua);
                attrSel.setEvaluator(evalInfo);
                attrSel.setSearch(ranker);
                attrSel.setInputFormat(dataBektorea);
                dataBektorea = Filter.useFilter(dataBektorea, attrSel);

                System.out.println("\n========== 4. FASEA: ATRIBUTUEN HAUTAKETA ==========");
                System.out.println("InfoGain ondoren hautatutako atributuak: " + dataBektorea.numAttributes());
                System.out.println("====================================================\n");
            }

            // ── 5, 6 eta 7. 10-fold cross-validation (C-ren doikuntza eta Ebaluazioa) ────
            System.out.println("\n========== 5 & 6. FASEAK: SVM-REN EBALUAZIOA ETA DOIKUNTZA ==========");
            double[] cBalioak = { 0.1, 0.5, 1.0, 2.0, 5.0 };
            double cOnena = 1.0;
            double accuracyOnena = -1.0;

            for (double c : cBalioak) {
                SMO smoTemp = new SMO();
                smoTemp.setC(c);
                Evaluation evalTemp = new Evaluation(dataBektorea);
                evalTemp.crossValidateModel(smoTemp, dataBektorea, 10, new Random(1));

                double accuracy = evalTemp.pctCorrect();
                double fMeasure = evalTemp.weightedFMeasure();
                System.out.println("-> SMO (C=" + c + ") | Accuracy: " + String.format("%.2f%%", accuracy)
                        + " | Weighted F-Measure: " + String.format("%.4f", fMeasure));

                if (accuracy > accuracyOnena) {
                    accuracyOnena = accuracy;
                    cOnena = c;
                }
            }
            System.out.println("\n*** Aurkitutako C parametro onena: " + cOnena + " ***");
            System.out.println("============================================================\n");

            // Probatutako eredu onenaren xehetasun zehatzak bakarrik inprimatu:
            SMO smo = new SMO();
            smo.setC(cOnena);
            Evaluation evalBest = new Evaluation(dataBektorea);
            evalBest.crossValidateModel(smo, dataBektorea, 10, new Random(1));
            System.out.println(evalBest.toSummaryString("Emaitz globalak (Eredu optimoa C=" + cOnena + ")", false));
            System.out.println(evalBest.toClassDetailsString("Klaseko xehetasunak"));
            System.out.println(evalBest.toMatrixString("Matrizea"));

            // ── 7. FASEA: Eredu osoarekin entrenatu (Datu guztiekin entrenatu) ────
            smo.buildClassifier(dataBektorea);

            // ── 6. TEST DATA kargatu ──────────────────────────────────────────
            File cleanTest = CSVPreprocessor.preprocessCSV(csvTestPath, false); // false errenkada itsuak ustekabean ez
                                                                                // ezabatzeko
            CSVLoader testLoader = new CSVLoader();
            testLoader.setStringAttributes("5"); // Testeko tweetak String gisa irakurtzera behartzen du
            testLoader.setSource(cleanTest);
            Instances testData = testLoader.getDataSet();

            // Test instanceak training-aren egiturarekin berreraiki
            Instances testEgitura = new Instances(dataFiltratuta, 0);
            for (int i = 0; i < testData.numInstances(); i++) {
                double[] vals = new double[dataFiltratuta.numAttributes()];

                // 1. Topic (en testData es la columna 0)
                String topicVal = testData.instance(i).stringValue(0);
                vals[0] = dataFiltratuta.attribute(0).indexOfValue(topicVal);
                if (vals[0] < 0)
                    vals[0] = Utils.missingValue();

                // 2. Sentiment (en testData es la 1)
                if (testData.instance(i).isMissing(1)) {
                    vals[1] = Utils.missingValue();
                } else {
                    String sentimentVal = testData.instance(i).stringValue(1);
                    vals[1] = dataFiltratuta.attribute(1).indexOfValue(sentimentVal);
                    if (vals[1] < 0) {
                        vals[1] = Utils.missingValue();
                    }
                }

                // 3. TweetText (en testData es la 4, ya que no hemos aplicado removeFilter)
                String textVal = testData.instance(i).stringValue(4);
                vals[2] = testEgitura.attribute(2).addStringValue(textVal);

                testEgitura.add(new DenseInstance(1.0, vals));
            }
            testEgitura.setClassIndex(1);

            Instances testBektorea = Filter.useFilter(testEgitura, STWV);

            if (atributuakOptimizatu) {
                testBektorea = Filter.useFilter(testBektorea, attrSel);
            }

            // ── 7. TEST DATA ebaluatu (Matrizea inprimatzeko) ─────────────────
            System.out.println("\n========== 7. FASEA: TEST DATUEN EBALUAZIOA ==========");
            Evaluation evalTest = new Evaluation(dataBektorea);
            evalTest.evaluateModel(smo, testBektorea);
            System.out.println(evalTest.toSummaryString("Test Emaitz globalak", false));
            System.out.println(evalTest.toClassDetailsString("Test Klaseko xehetasunak"));
            System.out.println(evalTest.toMatrixString("Test Matrizea"));
            System.out.println("========================================================\n");

            // ── 8. Iragarpenak idatzi ─────────────────────────────────────────
            try (FileWriter f = new FileWriter(outputPath)) {
                for (int i = 0; i < testBektorea.numInstances(); i++) {
                    double pred = smo.classifyInstance(testBektorea.instance(i));
                    String predLabel = dataBektorea.classAttribute().value((int) pred);
                    f.write(predLabel + "\n");
                }
            }
            System.out.println("Iragarpenak idatzita: " + outputPath);

            // Jatorrizko kontsola leheneratu prozesua amaitzean
            System.out.flush();
            System.setOut(jatorrizkoKontsola);
            jatorrizkoKontsola.println("[+] PROZESUA ARRAKASTATSU AMAITU DA! Berrikusi testu fitxategia: " + logPath);

        } catch (Exception e) {
            System.setOut(jatorrizkoKontsola);
            e.printStackTrace();
        }
    }
}