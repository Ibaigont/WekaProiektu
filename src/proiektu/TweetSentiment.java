package proiektu;

import java.io.File;
import java.io.FileWriter;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Random;

import weka.filters.Filter;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.DenseInstance;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.ClassBalancer;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;

/**
 * Pipeline optimizatua tweet sentimenduen sailkapenerako.
 *
 * Hobekuntza nagusiak (ClassBalancer berrezarrita):
 * 1. ClassBalancer (pisuak): CV-n egonkorra eta ezaguna den estrategia
 * 2. Grid Search: InfoGain atributu-kopurua × C parametroa bateratua
 * 3. NLP: hiztegia 5000, minFreq=2 (hapax kendu), L2 normalizazioa
 * 4. PolyKernel lineal (NLPrako gomendatua)
 */
public class TweetSentiment {

    // ── Konfigurazio nagusia ──────────────────────────────────────────────────
    private static final boolean INFOGAIN_SEARCH = true; // true → atributu-kopurua ere optimizatu
    private static final boolean KERNEL_SEARCH = false; // true → poly exp=1 vs exp=2 probatu
    private static final int CV_FOLDS = 10;
    private static final int RANDOM_SEED = 42;

    // StringToWordVector parametroak
    private static final int HIZTEGIA = 5000; // Hiztegi handiagoa, InfoGain-ek filtratuko du
    private static final int MIN_FREQ = 2; // Hapax kendu → generalizazio hobea
    private static final boolean BIGRAMAK = true;
    private static final boolean STEMMER = true;
    private static final boolean STOPWORDS = false; // ERAGINKORRA: Rainbow-k "not","never" kentzen ditu → sentimendua
                                                    // galtzen da
    private static final boolean NORMALIZE_L2 = true; // L2 normalizazioa SVM-rentzat ezinbestekoa

    public static void main(String[] args) {
        String csvPath = "data/tweetSentiment.train.csv";
        String csvTestPath = "data/tweetSentiment.dev.csv";
        String outputPath = "Iragarpenak/iragarpenak.txt";
        String logPath = "Iragarpenak/informe_resultados.txt";

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

        PrintStream konsola = System.out;
        try {
            System.setOut(new PrintStream(new FileOutputStream(logPath)));
            konsola.println("\n[+] Pipeline optimizatua exekutatzen (ClassBalancer + L2 + GridSearch)...");
            konsola.println("[+] Emaitzak: " + logPath);

            // ════════════════════════════════════════════════════════════════
            // 1. FASEA: Training data kargatu
            // ════════════════════════════════════════════════════════════════
            File cleanTrain = CSVPreprocessor.preprocessCSV(csvPath, true);
            CSVLoader loader = new CSVLoader();
            loader.setStringAttributes("5");
            loader.setSource(cleanTrain);
            Instances data = loader.getDataSet();

            // TweetId eta TweetDate kendu
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndices("3,4");
            removeFilter.setInputFormat(data);
            Instances dataFiltratuta = Filter.useFilter(data, removeFilter);
            // Orain: Topic(0) Sentiment(1) TweetText(2)
            dataFiltratuta.setClassIndex(1);
            dataFiltratuta.deleteWithMissingClass();

            System.out.println("========== 1. FASEA: DATU-SORTAREN ANALISIA ==========");
            System.out.println("Instantzia kopurua (Training): " + dataFiltratuta.numInstances());
            System.out.println("\nKlaseen banaketa:");
            int[] counts = dataFiltratuta.attributeStats(dataFiltratuta.classIndex()).nominalCounts;
            for (int i = 0; i < dataFiltratuta.numClasses(); i++) {
                System.out.printf("  %-12s : %d instantzia%n",
                        dataFiltratuta.classAttribute().value(i), counts[i]);
            }

            // OHARRA: ClassBalancer InfoGain ONDOREN aplikatuko da
            // (InfoGainAttributeEval-ek
            // ez ditu onartzen pisuak ez diren 1.0 diren instantziak).
            System.out.println("======================================================\n");

            // ════════════════════════════════════════════════════════════════
            // 2. FASEA: StringToWordVector (training datuekin fitatu)
            // ════════════════════════════════════════════════════════════════
            // STWV: dataFiltratuta (pisuak=1.0) erabiltzen da InfoGain-erako
            StringToWordVector stwv = eraikitzaileSTWV();
            stwv.setInputFormat(dataFiltratuta);
            Instances dataBektorea = Filter.useFilter(dataFiltratuta, stwv);

            // L2 normalizazioa: SVM hiperplanoa estabilizatzen du pisu handiko vektoreekiko
            Normalize normalizeFilter = null;
            if (NORMALIZE_L2) {
                normalizeFilter = new Normalize();
                normalizeFilter.setInputFormat(dataBektorea);
                dataBektorea = Filter.useFilter(dataBektorea, normalizeFilter);
            }

            System.out.println("========== 2. FASEA: STWV ONDOREN ==========");
            System.out.println("Atributu kopurua STWV ondoren: " + dataBektorea.numAttributes());
            System.out.println("(L2 normalizazioa: " + NORMALIZE_L2 + ", minFreq=" + MIN_FREQ + ")");
            System.out.println("==========================================\n");

            // ════════════════════════════════════════════════════════════════
            // 3. FASEA: Grid Search — InfoGain atributu-kopurua × C
            // ════════════════════════════════════════════════════════════════
            System.out.println("========== 3. FASEA: GRID SEARCH (InfoGain × C) ==========");

            // Grid parametroak
            int[] atributuKopuruak = INFOGAIN_SEARCH
                    ? new int[] { 500, 1000, 1500, 2000, -1 } // -1 = guztiak
                    : new int[] { -1 };
            double[] cBalioak = { 0.05, 0.1, 0.2, 0.5, 1.0 };

            double onenaScore = -1.0; // Accuracy-z optimizatzen dugu
            int onenaAtributuK = -1;
            double onenaC = 0.1;

            // InfoGain ranking osoa kalkulatu behin (efizientzia)
            AttributeSelection attrSelFull = new AttributeSelection();
            attrSelFull.setEvaluator(new InfoGainAttributeEval());
            Ranker rankerFull = new Ranker();
            rankerFull.setNumToSelect(-1);
            attrSelFull.setSearch(rankerFull);
            attrSelFull.setInputFormat(dataBektorea);
            Instances dataBektoreaRanked = Filter.useFilter(dataBektorea, attrSelFull);
            System.out.println("InfoGain ranking kalkulatuta. Atributu guztiak: "
                    + dataBektoreaRanked.numAttributes() + "\n");

            for (int nAttr : atributuKopuruak) {
                // 1) Atributu-hautaketa (InfoGain pisurik gabeko datuetan)
                Instances datuakInfoGain;
                if (nAttr == -1 || nAttr >= dataBektoreaRanked.numAttributes()) {
                    datuakInfoGain = dataBektoreaRanked;
                } else {
                    AttributeSelection attrSelN = new AttributeSelection();
                    attrSelN.setEvaluator(new InfoGainAttributeEval());
                    Ranker rankerN = new Ranker();
                    rankerN.setNumToSelect(nAttr);
                    attrSelN.setSearch(rankerN);
                    attrSelN.setInputFormat(dataBektorea);
                    datuakInfoGain = Filter.useFilter(dataBektorea, attrSelN);
                }

                // 2) ClassBalancer ORAIN aplikatu (InfoGain-ek jadanik egin du bere lana)
                ClassBalancer balGrid = new ClassBalancer();
                balGrid.setInputFormat(datuakInfoGain);
                Instances datuakGrid = Filter.useFilter(datuakInfoGain, balGrid);

                for (double c : cBalioak) {
                    SMO smoGrid = eraikitzaileSMO(c, 1.0);
                    Evaluation evalGrid = new Evaluation(datuakGrid);
                    evalGrid.crossValidateModel(smoGrid, datuakGrid, CV_FOLDS, new Random(RANDOM_SEED));

                    double acc = evalGrid.pctCorrect();
                    double wF1 = evalGrid.weightedFMeasure();
                    System.out.printf("  nAttr=%4s | C=%.2f | Accuracy=%.2f%% | Weighted F1=%.4f%n",
                            nAttr == -1 ? "guzt" : String.valueOf(nAttr), c, acc, wF1);

                    if (acc > onenaScore) {
                        onenaScore = acc;
                        onenaAtributuK = nAttr;
                        onenaC = c;
                    }
                }
            }

            System.out.printf("%n*** ONENA: nAttr=%s | C=%.2f | CV Accuracy=%.2f%% ***%n",
                    onenaAtributuK == -1 ? "guztiak" : String.valueOf(onenaAtributuK),
                    onenaC, onenaScore);
            System.out.println("==========================================================\n");

            // ════════════════════════════════════════════════════════════════
            // 4. FASEA: Kernel bilaketa (aukerazkoa)
            // ════════════════════════════════════════════════════════════════
            if (KERNEL_SEARCH) {
                // Onena aurkitutako nAttr-ekin iragazkia prestatu
                Instances datuakKernel = preparatuDatuak(dataBektorea, onenaAtributuK);
                System.out.println("========== KERNEL BILAKETA ==========");
                for (double exp : new double[] { 1.0, 2.0 }) {
                    SMO smoK = eraikitzaileSMO(onenaC, exp);
                    Evaluation evalK = new Evaluation(datuakKernel);
                    evalK.crossValidateModel(smoK, datuakKernel, CV_FOLDS, new Random(RANDOM_SEED));
                    System.out.printf("  Poly(exp=%.0f) | C=%.2f | Accuracy=%.2f%% | F1=%.4f%n",
                            exp, onenaC, evalK.pctCorrect(), evalK.weightedFMeasure());
                }
                System.out.println("=====================================\n");
            }

            // ════════════════════════════════════════════════════════════════
            // 5. FASEA: Eredu onena eraiki eta ebaluatu CV xehetasunez
            // ════════════════════════════════════════════════════════════════
            System.out.println("========== 5. FASEA: EREDU ONENA ENTRENATU ==========");

            // Atributu-hautaketa finala (InfoGain pisurik gabeko datuetan)
            AttributeSelection attrSelFinal = null;
            Instances datuakFinalak; // pisurik gabeko bertsioa → testrako erreferentzia
            Instances datuakFinalakOrekat; // ClassBalancer-ekin → CV eta entrenamendu finalerako

            if (onenaAtributuK != -1 && onenaAtributuK < dataBektoreaRanked.numAttributes()) {
                attrSelFinal = new AttributeSelection();
                attrSelFinal.setEvaluator(new InfoGainAttributeEval());
                Ranker rankerFinal = new Ranker();
                rankerFinal.setNumToSelect(onenaAtributuK);
                attrSelFinal.setSearch(rankerFinal);
                attrSelFinal.setInputFormat(dataBektorea);
                datuakFinalak = Filter.useFilter(dataBektorea, attrSelFinal);
            } else {
                datuakFinalak = dataBektoreaRanked;
                attrSelFinal = attrSelFull;
            }

            // ClassBalancer: InfoGain ONDOREN aplikatu
            ClassBalancer balFinal = new ClassBalancer();
            balFinal.setInputFormat(datuakFinalak);
            datuakFinalakOrekat = Filter.useFilter(datuakFinalak, balFinal);

            // CV xehetasunak eredu onenarekin (pisuak dituen bertsioarekin)
            SMO smoOnena = eraikitzaileSMO(onenaC, 1.0);
            Evaluation evalOnena = new Evaluation(datuakFinalakOrekat);
            evalOnena.crossValidateModel(smoOnena, datuakFinalakOrekat, CV_FOLDS, new Random(RANDOM_SEED));

            System.out.println(evalOnena.toSummaryString(
                    "Eredu optimoaren CV emaitzak (C=" + onenaC + ", nAttr=" +
                            (onenaAtributuK == -1 ? "guztiak" : onenaAtributuK) + ")",
                    false));
            System.out.println(evalOnena.toClassDetailsString("Klaseko xehetasunak (CV)"));
            System.out.println(evalOnena.toMatrixString("Nahasmendu-matrizea (CV)"));

            // Eredu osoa entrenatu datu guztiekin (pisuak dituen datu-sorta)
            SMO smoFinal = eraikitzaileSMO(onenaC, 1.0);
            smoFinal.buildClassifier(datuakFinalakOrekat);

            // ════════════════════════════════════════════════════════════════
            // 6. FASEA: Test data kargatu eta ebaluatu
            // ════════════════════════════════════════════════════════════════
            File cleanTest = CSVPreprocessor.preprocessCSV(csvTestPath, false);
            CSVLoader testLoader = new CSVLoader();
            testLoader.setStringAttributes("5");
            testLoader.setSource(cleanTest);
            Instances testData = testLoader.getDataSet();

            // Test instantziak training-aren egiturarekin berreraiki (leakage gabe)
            Instances testEgitura = new Instances(dataFiltratuta, 0);
            for (int i = 0; i < testData.numInstances(); i++) {
                double[] vals = new double[dataFiltratuta.numAttributes()];

                String topicVal = testData.instance(i).stringValue(0);
                vals[0] = dataFiltratuta.attribute(0).indexOfValue(topicVal);
                if (vals[0] < 0)
                    vals[0] = Utils.missingValue();

                if (testData.instance(i).isMissing(1)) {
                    vals[1] = Utils.missingValue();
                } else {
                    String sentimentVal = testData.instance(i).stringValue(1);
                    vals[1] = dataFiltratuta.attribute(1).indexOfValue(sentimentVal);
                    if (vals[1] < 0)
                        vals[1] = Utils.missingValue();
                }

                String textVal = testData.instance(i).stringValue(4);
                vals[2] = testEgitura.attribute(2).addStringValue(textVal);
                testEgitura.add(new DenseInstance(1.0, vals));
            }
            testEgitura.setClassIndex(1);

            // Test datuen transformazioa (filter berdinak training-etik, hurrenkera
            // garrantzitsua)
            // OHARRA: ClassBalancer EZ da aplikatzen test datuetan (pisuak training-erako
            // soilik)
            Instances testBektorea = Filter.useFilter(testEgitura, stwv);
            if (NORMALIZE_L2 && normalizeFilter != null) {
                testBektorea = Filter.useFilter(testBektorea, normalizeFilter);
            }
            testBektorea = Filter.useFilter(testBektorea, attrSelFinal);

            System.out.println("========== 6. FASEA: TEST DATUEN EBALUAZIOA ==========");
            // Erreferentzia: pisurik gabeko datuak (test emaitzak ez dira ponderatu behar)
            Evaluation evalTest = new Evaluation(datuakFinalak);
            evalTest.evaluateModel(smoFinal, testBektorea);
            System.out.println(evalTest.toSummaryString("Test emaitzak", false));
            System.out.println(evalTest.toClassDetailsString("Test klaseko xehetasunak"));
            System.out.println(evalTest.toMatrixString("Test nahasmendu-matrizea"));
            System.out.println("=======================================================\n");

            // ════════════════════════════════════════════════════════════════
            // 7. FASEA: Iragarpenak idatzi
            // ════════════════════════════════════════════════════════════════
            try (FileWriter fw = new FileWriter(outputPath)) {
                for (int i = 0; i < testBektorea.numInstances(); i++) {
                    double pred = smoFinal.classifyInstance(testBektorea.instance(i));
                    String predLabel = datuakFinalak.classAttribute().value((int) pred);
                    fw.write(predLabel + "\n");
                }
            }
            System.out.println("Iragarpenak idatzita: " + outputPath);

            System.out.flush();
            System.setOut(konsola);
            konsola.println("[+] PROZESUA ARRAKASTATSU AMAITU DA! Berrikusi: " + logPath);

        } catch (Exception e) {
            System.setOut(konsola);
            e.printStackTrace();
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Laguntzaile metodoak
    // ════════════════════════════════════════════════════════════════════════

    /**
     * StringToWordVector konfiguratu parametro optimizatuekin.
     * Hobekuntzak: hiztegia=5000, minFreq=2 (hapax kendu), L2norm.
     */
    private static StringToWordVector eraikitzaileSTWV() throws Exception {
        StringToWordVector stwv = new StringToWordVector();
        stwv.setAttributeIndices("last");
        stwv.setWordsToKeep(HIZTEGIA);
        stwv.setLowerCaseTokens(true);
        stwv.setTFTransform(true);
        stwv.setIDFTransform(true);
        stwv.setMinTermFreq(MIN_FREQ); // Hapax legomena kendu: hitz bakarrean agertzen direnak zarata dira
        stwv.setOutputWordCounts(false); // TF-IDF-rekin ez behar word counts

        if (BIGRAMAK) {
            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(1);
            tokenizer.setNGramMaxSize(2); // Bigrama: unigramak + bigramak (trigramak ez dira hoberik)
            stwv.setTokenizer(tokenizer);
        }
        if (STEMMER) {
            stwv.setStemmer(new IteratedLovinsStemmer());
        }
        if (STOPWORDS) {
            stwv.setStopwordsHandler(new Rainbow());
        }
        return stwv;
    }

    /**
     * SMO konfiguratu PolyKernel-ekin.
     * exp=1.0 → lineal (NLPrako gomendatua, espazio altua eta sparse da).
     * exp=2.0 → kuadratikoa (KERNEL_SEARCH=true denean probatu).
     */
    private static SMO eraikitzaileSMO(double c, double polyExponent) throws Exception {
        SMO smo = new SMO();
        smo.setC(c);

        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(polyExponent);
        kernel.setUseLowerOrder(polyExponent > 1.0);
        smo.setKernel(kernel);

        return smo;
    }

    /**
     * Datuak InfoGain-ekin iragazteko laguntzailea (KERNEL_SEARCH-erako).
     */
    private static Instances preparatuDatuak(Instances base, int nAttr) throws Exception {
        if (nAttr == -1 || nAttr >= base.numAttributes()) {
            AttributeSelection as = new AttributeSelection();
            as.setEvaluator(new InfoGainAttributeEval());
            Ranker r = new Ranker();
            r.setNumToSelect(-1);
            as.setSearch(r);
            as.setInputFormat(base);
            return Filter.useFilter(base, as);
        }
        AttributeSelection as = new AttributeSelection();
        as.setEvaluator(new InfoGainAttributeEval());
        Ranker r = new Ranker();
        r.setNumToSelect(nAttr);
        as.setSearch(r);
        as.setInputFormat(base);
        return Filter.useFilter(base, as);
    }
}