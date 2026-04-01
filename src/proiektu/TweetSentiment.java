package proiektu;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import weka.core.Instances;
import weka.filters.MultiFilter;

/**
 * Proiektuaren klase nagusia (Main). Exekuzio-fluxu osoa koordinatzen du:
 * datuak kargatu, garbitu, iragazkiak ezarri, eredu bat entrenatu eta ebaluatu, 
 * eta azkenik, datu berrien gaineko iragarpenak egiten ditu.
 * * @version 1.0
 */
public class TweetSentiment {

    /**
     * Aplikazioaren sarrera puntua. Esperimentuaren parametroak definitzen ditu,
     * pipeline osoa exekutatzen du eta emaitzak txosten batean gordetzen ditu.
     *
     * @param args Komando-lerroko argumentuak. Hurrenez hurren jaso ditzake: 
     * Train datuen bidea, Test datuen bidea, Iragarpenen irteera bidea eta Txostenaren (log) bidea.
     */
    public static void main(String[] args) {
        // --- Experimentatzeko balioak ---
        boolean stemmerErabili  = true;
        boolean stopWordsErabili = true;
        boolean bigramakErabili = true;
        int hiztegiTamaina = 1000;          
        boolean atributuakOptimizatu = false;
        int azkenAtributuKopurua = 300;    
        boolean datuakOrekatzeko = false;    
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

            // 1. Train datuak kargatu eta garbitu (GORDINAK dira)
            File cleanTrain = CSVPreprocessor.preprocessCSV(csvPath, true);
            Instances baseTrainData = DatasetManager.loadTrainData(cleanTrain);

            // 2. Iragazkien paketea sortu (Datuak ez dira hemen aldatzen)
            TextPipeline pipeline = new TextPipeline();
            MultiFilter multiFilter = pipeline.getMultiFilter(
                stemmerErabili, stopWordsErabili, bigramakErabili, hiztegiTamaina, atributuakOptimizatu, azkenAtributuKopurua, datuakOrekatzeko
            );

            // 3. Eredua entrenatu eta ebaluatu (FilteredClassifier-ak dena egingo du barruan)
            ModelManager modelManager = new ModelManager();
            // Datu gordinak pasatzen dizkiogu, iragazkien paketearekin batera
            modelManager.trainAndEvaluate(baseTrainData, multiFilter);
            
            // Eredua emaitzak karpetan gorde
            modelManager.saveModel("Emaitzak/eredua.model");

            // 4. Test datuak kargatu eta egitura berreraiki
            File cleanTest = CSVPreprocessor.preprocessCSV(csvTestPath, false);
            Instances baseTestData = DatasetManager.buildTestData(cleanTest, baseTrainData);

            // 5. Iragarpenak sortu
            // FilteredClassifier-ak berak modu automatikoan aplikatuko dizkio iragazkiak test datuei
            modelManager.testAndPredict(baseTestData, baseTrainData, outputPath);

            System.out.flush();
            System.setOut(jatorrizkoKontsola);
            jatorrizkoKontsola.println("[+] PROZESUA ARRAKASTATSU AMAITU DA! Berrikusi testu fitxategia: " + logPath);

        } catch (Exception e) {
            System.setOut(jatorrizkoKontsola);
            e.printStackTrace();
        }
    }
}