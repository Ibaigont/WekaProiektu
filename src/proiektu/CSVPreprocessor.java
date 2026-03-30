package proiektu;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * CSV fitxategiak kargatu aurretik garbitzeko eta prestatzeko erabiltzen den klase laguntzailea.
 * Bereziki, txioak (tweets) dituzten fitxategietako formatu-erroreak (adibidez, komatxo arteko lerro-jauziak)
 * konpontzen ditu, eta nahi izanez gero, maiztasun txikiko klaseak edo "klase fantasmak" baztertzen ditu.
 * * @version 1.0
 */
public class CSVPreprocessor {

    /**
     * Emandako bideko CSV fitxategia irakurtzen du eta garbiketa-prozesu bat aplikatzen dio.
     * Honako ekintza hauek burutzen ditu:
     * <ul>
     * <li>Komatxo arteko lerro-jauziak zuriune bihurtzen ditu, Weka-k ondo irakur dezan.</li>
     * <li>Komatxo arteko "" barne-komatxoak '' (komatxo sinple) bihurtzen ditu.</li>
     * <li>Bost zutabe ez dituzten errenkadak baztertzen ditu (formatu okerra dutelako).</li>
     * <li>Twitterreko zarata garbitzen du testuetatik (URLak, @ aipamenak eta RT etiketak ezabatuz).</li>
     * <li>{@code filterInvalidClasses} egia bada, gutxienez 5 agerpen ez dituzten klaseak baztertzen ditu.</li>
     * </ul>
     * Azkenik, aldi baterako fitxategi (temporary file) garbi bat sortu eta itzultzen du.
     *
     * @param path Garbitu behar den jatorrizko CSV fitxategiaren bidea (path-a).
     * @param filterInvalidClasses {@code true} bada, 5 agerpen baino gutxiago dituzten sentimendu-klaseak iragaziko/ezabatuko ditu (normalean entrenamendu-datuetan soilik aktibatzen da).
     * @return Garbitutako datuak dituen aldi baterako {@link java.io.File} fitxategi berria.
     * @throws Exception Fitxategia irakurtzean edo idaztean Sarrera/Irteera (I/O) erroreren bat gertatzen bada.
     */
    public static File preprocessCSV(String path, boolean filterInvalidClasses) throws Exception {
        StringBuilder content = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            int c;
            while ((c = br.read()) != -1) {
                content.append((char) c);
            }
        }

        List<String[]> rows = new ArrayList<>();
        String[] currentRow = new String[5];
        int col = 0;
        boolean inQuotes = false;
        StringBuilder cell = new StringBuilder();

        for (int i = 0; i < content.length(); i++) {
            char c = content.charAt(i);

            if (c == '"') {
                if (inQuotes && i + 1 < content.length() && content.charAt(i + 1) == '"') {
                    cell.append('\'');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (c == ',' && !inQuotes) {
                if (col < 5) currentRow[col] = cell.toString();
                col++;
                cell.setLength(0);
            } else if ((c == '\n' || c == '\r') && !inQuotes) {
                if (col < 5) currentRow[col] = cell.toString();
                col++;
                if (col == 5) rows.add(currentRow.clone());
                if (c == '\r' && i + 1 < content.length() && content.charAt(i + 1) == '\n') {
                    i++;
                }
                currentRow = new String[5];
                col = 0;
                cell.setLength(0);
            } else if ((c == '\n' || c == '\r') && inQuotes) {
                cell.append(' ');
                if (c == '\r' && i + 1 < content.length() && content.charAt(i + 1) == '\n') {
                    i++;
                }
            } else {
                cell.append(c);
            }
        }
        if (col == 4) {
            currentRow[col] = cell.toString();
            rows.add(currentRow.clone());
        }

        File tmp = File.createTempFile("tweets_clean", ".csv");
        tmp.deleteOnExit();
        int good = 0, bad = 0;
        
        // --- AURREKO PAUSOA: Klase sendoak dinamikoki detektatu ---
        // Hiztegi bat sortzen dugu sentimendu bakoitza zenbat aldiz errepikatzen den kontatzeko
        java.util.Map<String, Integer> classFrequencies = new java.util.HashMap<>();
        if (filterInvalidClasses) {
            for (int r = 1; r < rows.size(); r++) { // Goiburua saihesten dugu (r=0)
                String[] row = rows.get(r);
                if (row.length > 1 && row[1] != null) {
                    String sentiment = row[1].trim(); // Jatorrizko letra larriak mantentzen ditugu
                    classFrequencies.put(sentiment, classFrequencies.getOrDefault(sentiment, 0) + 1);
                }
            }
        }

        try (PrintWriter pw = new PrintWriter(new FileWriter(tmp))) {
            for (int r = 0; r < rows.size(); r++) {
                String[] row = rows.get(r);
                if (r == 0) {
                    pw.println(toCsvLine(row));
                    continue;
                }
                boolean valid = true;
                for (int cIdx = 0; cIdx < row.length; cIdx++) {
                    if (row[cIdx] == null) { valid = false; break; }
                    // Twitterreko zarata garbitu
                    row[cIdx] = row[cIdx].replaceAll("(?i)https?://\\S+\\s?", "")
                                         .replaceAll("(?i)@\\w+\\s?", "")
                                         .replaceAll("(?i)\\bRT\\b\\s*", "")
                                         .trim();
                }
                if (!valid) { bad++; continue; }

                // --- KLASE FANTASMEN IRAGAZKI DINAMIKOA ---
                // Datu-sorta Training bada, fitxategi osoan 5 agerpen baino gutxiago dituzten klaseen errenkadak baztertzen ditugu
                if (filterInvalidClasses) {
                    String originalSentiment = row[1];
                    if (classFrequencies.getOrDefault(originalSentiment, 0) < 5) {
                        bad++;
                        continue;
                    }
                }

                pw.println(toCsvLine(row));
                good++;
            }
        }
        System.out.println(path + ": " + good + " onak, " + bad + " baztertuak");
        return tmp;
    }

    /**
     * Karaktere-kateen (String) array bat jasotzen du eta CSV formatuko lerro baliodun bat sortzen du.
     * Array-ko elementu bakoitza komatxo bikoitzen artean sartzen du eta komaz banatzen ditu.
     *
     * @param row CSV fitxategiko errenkada bat ordezkatzen duen String array-a (zutabeetako balioak).
     * @return Errenkadaren testu-irudikapena, CSV formatu estandarrean (adibidez: "balioa1","balioa2").
     */
    private static String toCsvLine(String[] row) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < row.length; i++) {
            if (i > 0) sb.append(',');
            sb.append('"').append(row[i] == null ? "" : row[i]).append('"');
        }
        return sb.toString();
    }
}