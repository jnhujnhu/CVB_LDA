import Core.CVB_LDA;
import Core.Perplexity;

/**
 * Created by Kevin on 08/05/2017.
 */
public class Testmain {
    public static void main(String[] args) {
        try{

            String Home_Dir = "/Users/Kevin/Desktop/Laboratory/Problem/TestCorpus/";

            //training
            CVB_LDA LDA = new CVB_LDA(1.01, 0.01, 50, Home_Dir + "CVB_Res/");
            LDA.initWordMap(Home_Dir + "wordmap.mp");
            LDA.initParameters(Home_Dir + "reuters.dat", 1);
            while(!LDA.iterateVariationalUpdate());

            //generate test theta
            CVB_LDA LDA_test = new CVB_LDA(1.01, 0.01, 50, Home_Dir + "CVB_Tes/");
            LDA_test.initWordMap(Home_Dir + "wordmap.mp");
            LDA_test.initParameters(Home_Dir + "trainset.dat", 0.5);
            LDA_test.givenPhi(Home_Dir + "CVB_Res/model_iter_500.phi");
            while(!LDA_test.iterateVariationalUpdate());

            //evaluate perplexity
            Perplexity perplexity = new Perplexity(50, LDA.getWordMap()
                    , Home_Dir + "CVB_Tes/model_iter_500.theta"
                    , Home_Dir + "CVB_Res/model_iter_500.phi"
                    , Home_Dir + "trainset.dat"
                    , 0.5);
            perplexity.importParameters();
            System.out.format("%d : %.15f \n", 500, perplexity.evaluatePerplexity());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
