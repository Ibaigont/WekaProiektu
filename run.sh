#!/bin/bash

# ==========================================
# WEKA PROIEKTUA EXEKUTATZEKO SCRIPT-A
# ==========================================

# --- 1. Aldagaien Konfigurazioa ---

# Weka liburutegiaren bidea (weka.jar fitxategia non dagoen).
# Proiektuaren barruko 'lib' karpetan egon behar da normalean.
WEKA_JAR="lib/weka.jar"

# Java klase konpilatuak (bitarrak) kokatuta dauden direktorioa.
BIN_DIR="bin"

# Puntu sarrera nagusia (main metodoa osatzen duen klasea).
MAIN_CLASS="proiektu.TweetSentiment"

# Eredua entrenatzeko erabiliko den fitxategi nagusia (train).
TRAIN_DATA="data/tweetSentiment.train.csv"

# Ereduaren aurreikuspenak probatzeko fitxategia (test edo dev).
TEST_DATA="data/tweetSentiment.dev.csv"

# Mezu bakoitzaren iragarpenak idatziko diren fitxategiko bidea.
OUTPUT_PREDICTIONS="Iragarpenak/iragarpenak.txt"

# Wekak itzulitako zehaztasun datuak eta estatistikak gordeko diren fitxategia.
LOG_RESULTS="Emaitzak/emaitzak.txt"


# --- 2. Hasierako Egiaztapenak ---

# Weka liburutegia fisikoki existitzen dela egiaztatzen dugu script-ak jarraitu baino lehen.
if [ ! -f "$WEKA_JAR" ]; then
    echo "ERROREA: Ez da aurkitu weka.jar bide honetan: $WEKA_JAR"
    echo "Mesedez, ziurtatu weka.jar liburutegia deskargatu eta 'lib' karpetan sartu duzula."
    exit 1
fi


# --- 3. Java Ingurunea Konfiguratzea (Moduluak) ---

# Java 9 bertsiotik aurrera (Java 21 bertsioan bereziki), Java-ren segurtasun arauek
# Weka-k erabiltzen dituen barne hausnarketa metodoak (reflection) oztopatzen dituzte.
# '--add-opens' baimenak emanez, Weka ondo egikaritzea ahalbidetzen dugu errore barik.
JVM_FLAGS="--add-opens java.base/java.lang=ALL-UNNAMED \
           --add-opens java.base/java.util=ALL-UNNAMED \
           --add-opens java.base/java.lang.reflect=ALL-UNNAMED \
           --add-opens java.base/java.io=ALL-UNNAMED"


# --- 4. Exekuzio Nagusia ---

echo "[+] Proiektua exekutatzen ari da..."
# Java abiarazten da aipatutako bandera (flags), fitxategi eta liburutegi guztiekin.
java $JVM_FLAGS -cp "$BIN_DIR:$WEKA_JAR" "$MAIN_CLASS" "$TRAIN_DATA" "$TEST_DATA" "$OUTPUT_PREDICTIONS" "$LOG_RESULTS"

# Jasotako kode-irteera (exit code) 0 bada, dena ondo joan dela esan nahi du.
if [ $? -eq 0 ]; then
    echo "[+] Exekuzioa arrakastatsu amaitu da."
    echo "[+] Emaitza orokorrak eta estatistikak hemen dituzu: $LOG_RESULTS"
    echo "[+] Aurreikusitako sentimenduak (iragarpenak) hemen: $OUTPUT_PREDICTIONS"
else
    echo "[-] Errore bat gertatu da proiektua exekutatzerakoan."
fi
