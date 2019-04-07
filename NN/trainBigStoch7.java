import java.util.Random;
import java.lang.Math;

import java.io.File;
import java.io.PrintWriter;
import java.util.Scanner;
import java.io.FileNotFoundException;

public class trainBigStoch7
{
    private static long startTime = System.currentTimeMillis();
    public static void main(String[] args) throws FileNotFoundException {
        //Training neural net
        int batch = 2;
        int input = 4;
        int hidden = 4;
        int output = 4;
        
        double epsilon = (double)0.001;
        
        double[][] label = new double[batch][output];
        double[][] trainData = new double[batch][input];
        
        double[][] hiddenBiases = new double[batch][hidden];
        double[][] hiddenWeights = new double[input][hidden];
        
        double[][] finalBiases = new double[batch][output];
        double[][] finalWeights = new double[hidden][output];
        
        rand(label);
        rand(trainData);
        
        rand(hiddenBiases);
        rand(hiddenWeights);
        
        rand(finalBiases);
        rand(finalWeights);
        
        double[][] gamma = new double[batch][output];
        double[][] beta = new double[batch][output];
        
        rand(gamma);
        rand(beta);
               
        double cost = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights, true, gamma, beta);
        System.out.println("*********************************************************************");

        double[][] betaGrad = clone(beta);
        for(int i=0; i<betaGrad.length; i++) {
            for(int j=0; j<betaGrad[0].length; j++) {
                double[][] beta1 = eps(beta,i,j,epsilon);
                double newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights, false, gamma, beta1);
            
                double[][] beta2 = eps(beta,i,j,-epsilon);
                double newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights, false, gamma, beta2);
            
                betaGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(betaGrad);
        System.out.println("*********************************************************************");
        
        double[][] gammaGrad = clone(gamma);
        for(int i=0; i<gammaGrad.length; i++) {
            for(int j=0; j<gammaGrad[0].length; j++) {
                double[][] gamma1 = eps(hiddenBiases,i,j,epsilon);
                double newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights, false, gamma1, beta);
            
                double[][] gamma2 = eps(hiddenBiases,i,j,-epsilon);
                double newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights, false, gamma2, beta);
            
                gammaGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(gammaGrad);
        System.out.println("*********************************************************************");
        
        double[][] hiddenBiasesGrad = clone(hiddenBiases);
        for(int i=0; i<hiddenBiasesGrad.length; i++) {
            for(int j=0; j<hiddenBiasesGrad[0].length; j++) {
                double[][] hiddenBiases1 = eps(hiddenBiases,i,j,epsilon);
                double newCost1 = networkCost(trainData, label, hiddenBiases1, hiddenWeights, finalBiases, finalWeights, false, gamma, beta);
            
                double[][] hiddenBiases2 = eps(hiddenBiases,i,j,-epsilon);
                double newCost2 = networkCost(trainData, label, hiddenBiases2, hiddenWeights, finalBiases, finalWeights, false, gamma, beta);
            
                hiddenBiasesGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(hiddenBiasesGrad);
        System.out.println("*********************************************************************");
        
        double[][] finalBiasesGrad = clone(finalBiases);
        for(int i=0; i<finalBiasesGrad.length; i++) {
            for(int j=0; j<finalBiasesGrad[0].length; j++) {
                double[][] finalBiases1 = eps(finalBiases,i,j,epsilon);
                double newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases1, finalWeights, false, gamma, beta);
            
                double[][] finalBiases2 = eps(finalBiases,i,j,-epsilon);
                double newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases2, finalWeights, false, gamma, beta);
            
                finalBiasesGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(finalBiasesGrad);
        System.out.println("*********************************************************************");
        
        double[][] hiddenWeightsGrad = clone(hiddenWeights);
        for(int i=0; i<hiddenWeightsGrad.length; i++) {
            for(int j=0; j<hiddenWeightsGrad[0].length; j++) {
                double[][] hiddenWeights1 = eps(hiddenWeights,i,j,epsilon);
                double newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights1, finalBiases, finalWeights, false, gamma, beta);
            
                double[][] hiddenWeights2 = eps(hiddenWeights,i,j,-epsilon);
                double newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights2, finalBiases, finalWeights, false, gamma, beta);
            
                hiddenWeightsGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(hiddenWeightsGrad);
        System.out.println("*********************************************************************");
        
        double[][] finalWeightsGrad = clone(finalWeights);
        for(int i=0; i<finalWeightsGrad.length; i++) {
            for(int j=0; j<finalWeightsGrad[0].length; j++) {
                double[][] finalWeights1 = eps(finalWeights,i,j,epsilon);
                double newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights1, false, gamma, beta);
            
                double[][] finalWeights2 = eps(finalWeights,i,j,-epsilon);
                double newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights2, false, gamma, beta);
            
                finalWeightsGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(finalWeightsGrad);
        System.out.println("*********************************************************************");


    }

    public static double networkCost(
    double[][] trainData, double[][] label,
    double[][] hiddenBiases, double[][] hiddenWeights,
    double[][] finalBiases, double[][] finalWeights,
    boolean print,
    double[][] gamma, double[][] beta) {
        //Training neural net
        int batch = 2;
        int input = 4;
        int hidden = 4;
        int output = 4;

        double[][] hiddenOutput = new double[batch][hidden];
        double[][] hiddenOutputSig = new double[batch][hidden];

        double[][] finalOutput = new double[batch][output];
        double[][] finalOutputSig = new double[batch][output];       
        
        double[][] hiddenBiasesGrad = new double[batch][hidden];
        double[][] hiddenWeightsGrad = new double[input][hidden];
        
        double[][] finalBiasesGrad = new double[batch][output];
        double[][] finalWeightsGrad = new double[hidden][output];
        
        //Training net
        hiddenOutput = mul(trainData,hiddenWeights,true,true);
        bias(hiddenOutput,hiddenBiases);
        hiddenOutputSig = sigmoid(hiddenOutput);
        finalOutput = mul(hiddenOutputSig,finalWeights,true,true); 
        bias(finalOutput,finalBiases);
        
        double[] u = mean(finalOutput);
        double[] s = std(finalOutput,u);
        double e = (double)0.1;        
        double[][] finalOutput2 = norm(finalOutput,u,s,e);
        double[][] finalOutput3 = add(hadMul(finalOutput2,gamma),beta);
        finalOutputSig = sigmoid(finalOutput3);

        //Gradients of weights and biases
        double[][] dBeta = hadMul(sub(finalOutputSig,label),dsigmoid(finalOutput3));
        double[][] dGamma = hadMul(dBeta,finalOutput2);
        finalBiasesGrad = JacMul(Jac(finalOutput),hadMul(dBeta,gamma));
        finalWeightsGrad = mul(hiddenOutputSig,finalBiasesGrad,false,true);
        
        hiddenBiasesGrad = hadMul(mul(finalBiasesGrad,finalWeights,true,false),dsigmoid(hiddenOutput));
        hiddenWeightsGrad = mul(trainData,hiddenBiasesGrad,false,true);
        
        if(print) {
            //Print gradients
            print(dBeta);
            System.out.println("*************************************");
            print(dGamma);
            System.out.println("*************************************");
            print((hiddenBiasesGrad));
            System.out.println("*************************************");
            print((finalBiasesGrad));
            System.out.println("*************************************");
            print(hiddenWeightsGrad);
            System.out.println("*************************************");
            print(finalWeightsGrad);
            System.out.println("*************************************");
        }

        //Return cost
        return cost(finalOutputSig,label);
    }

    public static double[] mean(double[][] M) {
        double[] u = new double[M[0].length];
        for(int j=0; j<M[0].length;j++) {
            double sum = 0;
            for(int i=0; i<M.length; i++) {
                sum += M[i][j];
            }
            u[j] = sum/M.length;
        }
        return u;
    }

    public static double[] std(double[][] M, double[] u) {
        double[] s = new double[M[0].length];
        for(int j=0; j<M[0].length;j++) {
            double sum = 0;
            for(int i=0; i<M.length; i++) {
                sum += (M[i][j] - u[j])*(M[i][j] - u[j]);
            }
            s[j] = sum/M.length;
        }
        return s;
    }
    
    public static double[][] norm(double[][] M, double[] u, double[] s, double e) {
        double[][] M2 = new double[M.length][M[0].length];
        for(int j=0; j<M[0].length;j++) {
            for(int i=0; i<M.length; i++) {
                M2[i][j] = (M[i][j] - u[j])/(double)Math.sqrt(s[j] + e);
            }
        }
        return M2;
    }
    
    public static double one(int i, int j) {
        if(i==j) {
            return 1;
        }
        
        else {
            return 0;
        }
    }
    
    public static double[][][] Jac(double[][] M) {
        double[][][] Jac = new double[M[0].length][M.length][M.length];
        //print(M);
        double[] u = mean(M);
        //print(u);
        double[] s = std(M,u);
        //print(s);
        double e = (double)0.1;
        
        double batch = (double)M.length;
        for(int c=0; c<M[0].length; c++) {
            for(int i=0; i<M.length; i++) {
                for(int j=0; j<M.length; j++) {
                    double sum = 0;
                    for(int k=0; k<M.length; k++) {
                        sum += 2*(M[k][c] - u[c])*(one(i,k)-(1/batch));
                    }
                    Jac[c][i][j] = (one(i,j) - (1/batch))*(double)Math.pow(s[c]+e,-0.5) + (double)(M[j][c] - u[c])*(double)(-0.5*Math.pow(s[c]+e,-1.5))*(double)((1/batch)*sum);
                }
            }
        }
        return Jac;
    }
        
    public static double[][] JacMul(double[][][] Jac, double[][] a) {        
        double[][] x = new double[a.length][a[0].length];
        for(int c=0; c<Jac.length; c++) {
            for(int i=0; i<Jac[0].length; i++) {
                for(int j=0; j<Jac[0][0].length; j++) {
                    x[i][c] += Jac[c][i][j]*a[j][c];
                }
            }
        }
        return x;
    }
    
    public static double[][] eps(double[][] a, int a1, int a2, double epsilon) {
        double[][] x = clone(a);
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                x[i][j] = a[i][j];
            }
        }
        x[a1][a2] = a[a1][a2] + epsilon;
        return x;
    }
    
    public static double[][] clone(double[][] a) {
        double[][] x = new double[a.length][a[0].length];
        return x;
    }
    
    public static void update(double a[][], double b[][], double l) {
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                a[i][j] -= l*b[i][j];
            }
        }
    }
    
    //http://www.i-programmer.info/programming/theory/2744-how-not-to-shuffle-the-kunth-fisher-yates-algorithm.html
    public static void sample(double lab[][], double labSamp[][], double train[][], double trainSamp[][], int b)
    {
        Random r = new Random();
        
        int a = lab.length;
        int[] index = new int[a];

        for(int i=0; i<b; i++)
        {
            int swap = r.nextInt(a-i)+i;
            
            int value1 = (index[swap] == 0) ? swap : index[swap];
            int value2 = (index[i] == 0) ? i : index[i];
            
            index[i] = value1;
            index[swap] = value2;
        }
                
        for(int k=0; k<b; k++)
        {
            for(int l=0; l<lab[0].length; l++)
            {
                labSamp[k][l] = lab[index[k]][l];
            }
        }
        
        for(int k=0; k<b; k++)
        {
            for(int l=0; l<train[0].length; l++)
            {
                trainSamp[k][l] = train[index[k]][l];
            }
        }
    }
    
    public static void bias(double a[][], double b[][])
    {        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                a[i][j] = a[i][j]+b[i][j];
            }
        }
    }
    
    public static double[][] mul(double[][] a, double[][] b, boolean left, boolean right) {
        int m1,n1;
        int m2,n2;
        
        double[][] c;
        
        if(left) {
            if(right) {
                m1 = a.length;
                n1 = a[0].length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new double[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[i][k]*b[k][j];
                        }
                    }
                }
            }
            
            else {
                m1 = a.length;
                n1 = a[0].length;
                m2 = b[0].length;
                n2 = b.length;
                
                c = new double[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[i][k]*b[j][k];
                        }
                    }
                }
            }
        }
        
        else {
            if(right) {
                m1 = a[0].length;
                n1 = a.length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new double[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[k][i]*b[k][j];
                        }
                    }
                }
            }
            
            else {
                m1 = a[0].length;
                n1 = a.length;
                m2 = b[0].length;
                n2 = b.length;
                
                c = new double[m1][n2];
                assert(n1 == m2): "wrong matrix dimension";
                for(int i=0; i<m1; i++) {
                    for(int j=0; j<n2; j++) {
                        for(int k=0; k<n1; k++) {
                            c[i][j] += a[k][i]*b[j][k];
                        }
                    }
                }
            }
        }

        return c;
    }
    
    public static void rand(double a[][])
    {
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                Random r = new Random();
                a[i][j] = (double)r.nextGaussian();
            }
        }
    }
    
    public static double[][] sigmoid(double a[][])
    {
        double[][] x = new double[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = (double)(1/(1+Math.exp(-a[i][j])));
            }
        }
        
        return x;
    }
    
    public static double[][] dsigmoid(double a[][])
    {
        double[][] x = new double[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                double sig = (double)(1/(1+Math.exp(-a[i][j])));
                x[i][j] = sig*(1-sig);
            }
        }
        
        return x;
    }

    public static double cost(double a[][], double b[][])
    {
        double x = 0;
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x+= (a[i][j] - b[i][j])*(a[i][j] - b[i][j]);
            }
        }
        
        x/= 2;
        return x;
    }

    public static double[][] hadMul(double a[][], double b[][])
    {
        double[][] x = new double[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]*b[i][j];
            }
        }
        
        return x;
    }
    
    public static double[][] sub(double a[][], double b[][])
    {
        double[][] x = new double[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]-b[i][j];
            }
        }
        
        return x;
    }
    
    public static double[][] add(double a[][], double b[][])
    {
        double[][] x = new double[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]+b[i][j];
            }
        }
        
        return x;
    }

    public static double[][] col(double a[][])
    {
        double[][] x = new double[1][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[0][j]+=a[i][j];
            }
        }
        
        return x;
    }
    
    public static void print(double[][][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                for(int k=0; k<a[0][0].length; k++) {
                    System.out.printf("%.8f ",a[i][j][k]);
                }
                System.out.println();
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(double[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                System.out.printf("%.8f ",a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
    
    public static void print(double[] a) {        
        for(int i=0; i<a.length; i++) {
            System.out.printf("%.8f ",a[i]);
        }
        System.out.println();
    }

    public static double[][] max(double a[][])
    {
        double[] max = new double[a.length];
        double[][] x = new double[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++)
        {
            double m=0;
            for(int j=0; j<a[0].length; j++)
            {
                if(m<a[i][j])
                {
                    m=a[i][j];
                    max[i] = j;
                }
            }
            x[i][(int)max[i]] = 1;
        }
        
        return x;
    }
    
    public static double accuracy(double a[][], double b[][])
    {
        double numerator = 0;
        double denominator = a.length;
        
        for(int i=0; i<a.length; i++)
        {
            double flag = 0;
            for(int j=0; j<a[0].length; j++)
            {
                if((a[i][j]==1) && (b[i][j]==1))
                {
                    flag=1;
                }
            }
            numerator+=flag;
        }

        return numerator/denominator;
    }
}