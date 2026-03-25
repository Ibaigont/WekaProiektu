package proiektu;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CSVPreprocessor {

    /**
     * CSV fitxategia irakurri eta garbitu:
     * - Komatxo arteko lerro-jauziak zuriune bihurtu
     * - Komatxo arteko "" barne-komatxoak '' bihurtu
     * - 5 zutabe ez dituzten errenkadak baztertu
     * Fitxategi garbi berri bat itzultzen du.
     */
    public static File preprocessCSV(String path) throws Exception {
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
        try (PrintWriter pw = new PrintWriter(new FileWriter(tmp))) {
            for (int r = 0; r < rows.size(); r++) {
                String[] row = rows.get(r);
                if (r == 0) {
                    pw.println(toCsvLine(row));
                    continue;
                }
                boolean valid = true;
                for (String val : row) {
                    if (val == null) { valid = false; break; }
                }
                if (!valid) { bad++; continue; }
                pw.println(toCsvLine(row));
                good++;
            }
        }
        System.out.println(path + ": " + good + " onak, " + bad + " baztertuak");
        return tmp;
    }

    private static String toCsvLine(String[] row) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < row.length; i++) {
            if (i > 0) sb.append(',');
            sb.append('"').append(row[i] == null ? "" : row[i]).append('"');
        }
        return sb.toString();
    }
}