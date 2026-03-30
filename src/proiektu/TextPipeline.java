package proiektu;

import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.instance.Resample;
import java.util.ArrayList;

/**
 * Testu-mezuak (txioak) prozesatzeko eta bektorizatzeko iragazki-katea (pipeline) 
 * konfiguratzeaz arduratzen den klasea. Weka-ren {@link MultiFilter} bat sortzen du, 
 * hainbat iragazki desberdin bakarrean bilduz.
 * * @version 1.0
 */
public class TextPipeline {
    
    /**
     * Datuen eraldaketa egingo duen iragazki anitza (MultiFilter) konfiguratzen eta itzultzen du.
     * Prozesuak hiru fase izan ditzake: testua hitz-bektore bihurtzea, atributuen optimizazioa eta datuen orekatzea.
     * * @param stemmer {@code true} bada, hitzen erroak ateratzeko (stemming) algoritmoa aplikatuko da.
     * @param stopWords {@code true} bada, esanahirik gabeko hitz arruntak (stop words) ezabatuko dira.
     * @param bigrams {@code true} bada, hitz bakuniez gain, ondoz ondoko bi hitzeko konbinazioak (bigramak) ere kontuan hartuko dira.
     * @param vocabSize Hiztegian gordeko den gehienezko hitz (edo n-grama) kopurua.
     * @param useAttrOptimization {@code true} bada, informazio gehien ematen duten atributuak bakarrik hautatuko dira (AttributeSelection).
     * @param numAttrs Hautatuko den atributu kopuru maximoa (optimizazioa aktibatuta badago).
     * @param balanceData {@code true} bada, klase minoritarioak gainlaginduko dira (Oversampling) datu-multzoa orekatzeko.
     * @return Konfiguratutako iragazki guztiak biltzen dituen {@link MultiFilter} objektua.
     * @throws Exception Iragazkien konfigurazioan erroreren bat gertatzen bada.
     */
    public MultiFilter getMultiFilter(boolean stemmer, boolean stopWords, boolean bigrams, int vocabSize, boolean useAttrOptimization, int numAttrs, boolean balanceData) throws Exception {
        
        ArrayList<Filter> iragazkiZerrenda = new ArrayList<>();

        // 1. StringToWordVector (Bektorizazioa)
        StringToWordVector stwv = new StringToWordVector();
        stwv.setAttributeIndices("last");
        stwv.setWordsToKeep(vocabSize);
        stwv.setLowerCaseTokens(true);
        stwv.setTFTransform(true);
        stwv.setIDFTransform(true);

        if (bigrams) {
            weka.core.tokenizers.NGramTokenizer tokenizer = new weka.core.tokenizers.NGramTokenizer();
            tokenizer.setNGramMinSize(1);
            tokenizer.setNGramMaxSize(2);
            stwv.setTokenizer(tokenizer);
        }
        if (stemmer) stwv.setStemmer(new weka.core.stemmers.IteratedLovinsStemmer());
        if (stopWords) stwv.setStopwordsHandler(new weka.core.stopwords.Rainbow());
        
        iragazkiZerrenda.add(stwv); // Lehenengo iragazkia gehitu

        // 2. AttributeSelection (Atributu hautaketa)
        if (useAttrOptimization) {
            AttributeSelection attrSel = new AttributeSelection();
            weka.attributeSelection.InfoGainAttributeEval evalInfo = new weka.attributeSelection.InfoGainAttributeEval();
            weka.attributeSelection.Ranker ranker = new weka.attributeSelection.Ranker();
            ranker.setNumToSelect(numAttrs);
            attrSel.setEvaluator(evalInfo);
            attrSel.setSearch(ranker);
            iragazkiZerrenda.add(attrSel); // Bigarren iragazkia gehitu
        }

        // 3. Resample (Datuak orekatzea)
        if (balanceData) {
            Resample resample = new Resample();
            resample.setBiasToUniformClass(1.0); 
            resample.setNoReplacement(false);    // Oversampling
            resample.setSampleSizePercent(150.0);
            iragazkiZerrenda.add(resample); // Hirugarren iragazkia gehitu
        }

        // Dena MultiFilter batean sartu eta itzuli
        MultiFilter multiFilter = new MultiFilter();
        multiFilter.setFilters(iragazkiZerrenda.toArray(new Filter[0]));
        
        return multiFilter;
    }
}