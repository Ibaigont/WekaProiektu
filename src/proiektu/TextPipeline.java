package proiektu;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;

public class TextPipeline {
    private StringToWordVector stwv;
    private AttributeSelection attrSel;
    private final boolean useAttrOptimization;

    public TextPipeline(boolean useAttrOptimization) {
        this.useAttrOptimization = useAttrOptimization;
    }

    public Instances applyTrainFilters(Instances data, boolean stemmer, boolean stopWords, boolean bigrams, int vocabSize, int numAttrs) throws Exception {
        stwv = new StringToWordVector();
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

        if (stemmer) {
            stwv.setStemmer(new weka.core.stemmers.IteratedLovinsStemmer());
        }

        if (stopWords) {
            stwv.setStopwordsHandler(new weka.core.stopwords.Rainbow());
        }

        stwv.setInputFormat(data);
        Instances vectorData = Filter.useFilter(data, stwv);

        if (useAttrOptimization) {
            attrSel = new AttributeSelection();
            weka.attributeSelection.InfoGainAttributeEval evalInfo = new weka.attributeSelection.InfoGainAttributeEval();
            weka.attributeSelection.Ranker ranker = new weka.attributeSelection.Ranker();
            ranker.setNumToSelect(numAttrs);
            attrSel.setEvaluator(evalInfo);
            attrSel.setSearch(ranker);
            attrSel.setInputFormat(vectorData);
            vectorData = Filter.useFilter(vectorData, attrSel);
            
            System.out.println("\n========== 4. FASEA: ATRIBUTUEN HAUTAKETA ==========");
            System.out.println("InfoGain ondoren hautatutako atributuak: " + vectorData.numAttributes());
            System.out.println("====================================================\n");
        }
        return vectorData;
    }

    public Instances applyTestFilters(Instances testData) throws Exception {
        Instances vectorTest = Filter.useFilter(testData, stwv);
        if (useAttrOptimization && attrSel != null) {
            vectorTest = Filter.useFilter(vectorTest, attrSel);
        }
        return vectorTest;
    }
}