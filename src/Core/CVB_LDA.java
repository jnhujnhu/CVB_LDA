package Core;

import Utils.Sampler;

import java.io.*;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

/**
 * Created by Kevin on 08/05/2017.
 */
public class CVB_LDA {

    private double Alpha, Beta;
    private int K, V, D, iter_No = 0;
    private Map<String, Integer> wordMap;
    private String outputDir;

    //variational parameters
    private double[][][] gamma;
    private double[][] mean_nkw;
    private double[] mean_nkd;
    private double[][] mean_jkd;

    private int[] doc_word;
    private int[][] word;
    private double[][] Phi = null;

    public CVB_LDA(double alpha, double beta, int K, String outputDir) {
        Alpha = alpha;
        Beta = beta;
        this.K = K;
        this.outputDir = outputDir;
        wordMap = new HashMap<>();
        V = 0;
    }

    public void initWordMap(String wordMapPath) throws FileNotFoundException {
        System.out.println("Init word map...");
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(wordMapPath))));
        Scanner scanner = new Scanner(input);
        V = Integer.parseInt(scanner.nextLine()); //word map No.
        while (scanner.hasNextLine()) {
            String[] temp = scanner.nextLine().split(" ");
            wordMap.put(temp[0], Integer.parseInt(temp[1]));
        }
        scanner.close();
    }

    public void initParameters(String DataDir, double doc_percent) throws Exception {
        System.out.println("Init parameters...");
        if(V == 0)
            throw new Exception("Word map not initialized!");
        BufferedReader input = new BufferedReader(new InputStreamReader(new FileInputStream(new File(DataDir))));
        Scanner scanner = new Scanner(input);
        String[] temp = scanner.nextLine().split(" ");

        D = Integer.parseInt(temp[1]); //doc count

        gamma = new double[D][][];
        mean_nkd = new double[K];
        mean_nkw = new double[K][V];
        mean_jkd = new double[D][K];

        doc_word = new int[D];
        word = new int[D][];

        int doc_no = 0;
        while(scanner.hasNextLine()) {
            String[] temp_doc = scanner.nextLine().split(" ");
            int doc_length = (int) Math.floor((temp_doc.length - 1) * doc_percent);

            doc_word[doc_no] = doc_length;
            word[doc_no] = new int[doc_length];
            gamma[doc_no] = new double[doc_length][K];
            for (int i = 0; i < doc_length; i++) {
                word[doc_no][i] = wordMap.get(temp_doc[i + 1]);
            }
            doc_no ++;
        }
        scanner.close();

        //random initialize
        for(int d = 0; d < D; d ++) {
            double[] theta;
            theta = Sampler.getDirichletSample(K, Alpha);
            double[] b_sigma = new double[K];
            for(int k2 = 0; k2 < K; k2 ++) {
                b_sigma[k2] = 0.5;
            }
            for(int n = 0; n < doc_word[d]; n ++) {
                gamma[d][n] = Sampler.getGaussianSample(K, theta, b_sigma);

                double gamma_norm = 0;
                for(int k = 0; k < K; k ++) {
                    gamma_norm += Math.exp(gamma[d][n][k]);
                }

                for(int k = 0; k < K; k ++) {
                    gamma[d][n][k] = Math.exp(gamma[d][n][k]) / gamma_norm;
                    mean_nkd[k] += gamma[d][n][k];
                    mean_jkd[d][k] += gamma[d][n][k];
                    mean_nkw[k][word[d][n] - 1] += gamma[d][n][k];
                }
            }
        }
    }

    private double mean_count_gamma(int ex_d, int ex_n, int k, int wsdn, int doc) {
        if(wsdn == 0 && doc == -1)
            return mean_nkd[k] - gamma[ex_d][ex_n][k];
        else if(doc == -1)
            return mean_nkw[k][wsdn - 1] - gamma[ex_d][ex_n][k];
        else
            return mean_jkd[doc][k] - gamma[ex_d][ex_n][k];
    }

    //CVB0
    public boolean iterateVariationalUpdate() throws Exception {
        iter_No ++;

        for(int d = 0; d < D; d ++) {
            for(int n = 0; n < doc_word[d]; n ++) {
                double norm = 0;
                double[] prev_gamma = new double[K];
                for(int k = 0; k < K; k ++) {
                    prev_gamma[k] = gamma[d][n][k];
                    if(Phi == null) {
                        gamma[d][n][k] = (Alpha + mean_count_gamma(d, n, k, 0, d))
                                * (Beta + mean_count_gamma(d, n, k, word[d][n], -1))
                                / (V * Beta + mean_count_gamma(d, n, k, 0, -1));
                    }
                    else {
                        gamma[d][n][k] = (Alpha + mean_count_gamma(d, n, k, 0, d))
                                * Phi[k][word[d][n] - 1];
                    }
                    norm += gamma[d][n][k];
                }
                for(int k = 0; k < K; k ++) {
                    gamma[d][n][k] /= norm;

                    //maintain mean_nkw mean_nkd
                    mean_jkd[d][k] += gamma[d][n][k] - prev_gamma[k];
                    mean_nkw[k][word[d][n] - 1] += gamma[d][n][k] - prev_gamma[k];
                    mean_nkd[k] += gamma[d][n][k] - prev_gamma[k];
                }
            }
        }
        System.out.println("Iterating... No: " + iter_No);
        if(iter_No == 500) {
            storePhi();
            storeTheta();
            return true;
        }
        return false;
    }

    public void givenPhi(String phiPath) throws Exception {
        Phi = new double[K][V];
        Perplexity.readFromFile(Phi, phiPath);
    }

    private void storePhi() throws Exception {
        PrintWriter writerphi = new PrintWriter(outputDir + "model_iter_" + iter_No + ".phi", "UTF-8");
        for(int k = 0; k < K; k ++) {
            for (int v = 0; v < V; v++)
                writerphi.print(Double.toString((Beta + mean_nkw[k][v]) / (V * Beta + mean_nkd[k])) + " ");
            writerphi.println();
        }
        writerphi.close();
    }

    private void storeTheta() throws Exception {
        PrintWriter writerphi = new PrintWriter(outputDir + "model_iter_" + iter_No + ".theta", "UTF-8");
        for(int d = 0; d < D; d ++) {
            double sum_jdd = 0;
            for(int k = 0; k < K; k ++)
                sum_jdd += mean_jkd[d][k];
            for(int k = 0; k < K; k ++) {
                writerphi.print(Double.toString((Alpha + mean_jkd[d][k]) / (K * Alpha + sum_jdd)) + " ");
            }
            writerphi.println();
        }
        writerphi.close();
    }

    public Map<String, Integer> getWordMap() {
        return wordMap;
    }
}
