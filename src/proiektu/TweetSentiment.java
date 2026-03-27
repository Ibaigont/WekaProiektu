package proiektu;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import weka.core.Instances;

public class TweetSentiment {

    public static void main(String[] args) {
        // --- Experimentatzeko balioak ---
        boolean stemmerErabili = true;
        boolean stopWordsErabili = true;
        boolean bigramakErabili = false;
        int hiztegiTamaina = 1000;
        boolean atributuakOptimizatu = true;
        int azkenAtributuKopurua = 300;
        // --------------------------------------------------

        String csvPath = "data/tweetSentiment.train.csv";
        String csvTestPath = "data/tweetSentiment.dev.csv";
        String outputPath = "Iragarpenak/iragarpenak.txt";
        String logPath = "Emaitzak/emaitzak.txt";

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
            jatorrizkoKontsola.println("[+] KONTUZ: Kontsola isilik egongo da, analisia gordetzen ari delako hemen: " + logPath + "\n");

            // 1. Train datuak kargatu eta garbitu
            File cleanTrain = CSVPreprocessor.preprocessCSV(csvPath, true);
            Instances baseTrainData = DatasetManager.loadTrainData(cleanTrain);

            // 2. Text Pipeline aplikatu (Train)
            TextPipeline pipeline = new TextPipeline(atributuakOptimizatu);
            Instances vectorTrainData = pipeline.applyTrainFilters(
                baseTrainData, stemmerErabili, stopWordsErabili, bigramakErabili, hiztegiTamaina, azkenAtributuKopurua
            );

            // 3. Eredua entrenatu eta ebaluatu (Cross-Validation)
            ModelManager modelManager = new ModelManager();
            modelManager.trainAndEvaluate(vectorTrainData);

            // 4. Test datuak kargatu eta egitura berreraiki
            File cleanTest = CSVPreprocessor.preprocessCSV(csvTestPath, false);
            Instances baseTestData = DatasetManager.buildTestData(cleanTest, baseTrainData);

            // 5. Text Pipeline aplikatu (Test datuetan)
            Instances vectorTestData = pipeline.applyTestFilters(baseTestData);

            // 6. Iragarpenak sortu
            modelManager.testAndPredict(vectorTestData, vectorTrainData, outputPath);

            System.out.flush();
            System.setOut(jatorrizkoKontsola);
            jatorrizkoKontsola.println("[+] PROZESUA ARRAKASTATSU AMAITU DA! Berrikusi testu fitxategia: " + logPath);

        } catch (Exception e) {
            System.setOut(jatorrizkoKontsola);
            e.printStackTrace();
        }
    }
}