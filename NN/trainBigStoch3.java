import java.util.Random;
import java.lang.Math;

import java.io.File;
import java.io.PrintWriter;
import java.util.Scanner;
import java.io.FileNotFoundException;

public class trainBigStoch3
{
    private static long startTime = System.currentTimeMillis();
    public static void main(String[] args) throws FileNotFoundException {
        //Training neural net
        int batch = 2;
        int input = 4;
        int hidden = 4;
        int output = 4;
        
        float epsilon = (float)0.001;
        
        float[][] label = new float[batch][output];
        float[][] trainData = new float[batch][input];
        
        float[][] hiddenBiases = new float[1][hidden];
        float[][] hiddenWeights = new float[input][hidden];
        
        float[][] finalBiases = new float[1][output];
        float[][] finalWeights = new float[hidden][output];
        
        rand(label);
        rand(trainData);
        
        rand(hiddenBiases);
        rand(hiddenWeights);
        
        rand(finalBiases);
        rand(finalWeights);
        
        float cost = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights, true);
        System.out.println("*********************************************************************");
        System.out.println("*********************************************************************");
        
        float[][] hiddenBiasesGrad = clone(hiddenBiases);
        for(int i=0; i<hiddenBiasesGrad[0].length; i++) {
            float[][] hiddenBiases1 = eps(hiddenBiases,0,i,epsilon);
            float newCost1 = networkCost(trainData, label, hiddenBiases1, hiddenWeights, finalBiases, finalWeights, false);
            
            float[][] hiddenBiases2 = eps(hiddenBiases,0,i,-epsilon);
            float newCost2 = networkCost(trainData, label, hiddenBiases2, hiddenWeights, finalBiases, finalWeights, false);
            
            hiddenBiasesGrad[0][i] = (newCost1 - newCost2)/(2*epsilon);
        }
        print(hiddenBiasesGrad);
        System.out.println("*********************************************************************");
        
        float[][] finalBiasesGrad = clone(finalBiases);
        for(int i=0; i<finalBiasesGrad[0].length; i++) {
            float[][] finalBiases1 = eps(finalBiases,0,i,epsilon);
            float newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases1, finalWeights, false);
            
            float[][] finalBiases2 = eps(finalBiases,0,i,-epsilon);
            float newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases2, finalWeights, false);
            
            finalBiasesGrad[0][i] = (newCost1 - newCost2)/(2*epsilon);
        }       
        print(finalBiasesGrad);
        System.out.println("*********************************************************************");

        float[][] hiddenWeightsGrad = clone(hiddenWeights);
        for(int i=0; i<hiddenWeightsGrad.length; i++) {
            for(int j=0; j<hiddenWeightsGrad[0].length; j++) {
                float[][] hiddenWeights1 = eps(hiddenWeights,i,j,epsilon);
                float newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights1, finalBiases, finalWeights, false);
            
                float[][] hiddenWeights2 = eps(hiddenWeights,i,j,-epsilon);
                float newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights2, finalBiases, finalWeights, false);
            
                hiddenWeightsGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(hiddenWeightsGrad);
        System.out.println("*********************************************************************");
        
        float[][] finalWeightsGrad = clone(finalWeights);
        for(int i=0; i<finalWeightsGrad.length; i++) {
            for(int j=0; j<finalWeightsGrad[0].length; j++) {
                float[][] finalWeights1 = eps(finalWeights,i,j,epsilon);
                float newCost1 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights1, false);
            
                float[][] finalWeights2 = eps(finalWeights,i,j,-epsilon);
                float newCost2 = networkCost(trainData, label, hiddenBiases, hiddenWeights, finalBiases, finalWeights2, false);
            
                finalWeightsGrad[i][j] = (newCost1 - newCost2)/(2*epsilon);
            }
        }
        print(finalWeightsGrad);
        System.out.println("*********************************************************************");       
    }
    
    public static float networkCost(float[][] trainData,float[][] label,float[][] hiddenBiases,float[][] hiddenWeights,float[][] finalBiases,float[][] finalWeights,boolean print) {
        //Training neural net
        int batch = 2;
        int input = 4;
        int hidden = 4;
        int output = 4;

        float[][] hiddenOutput = new float[batch][hidden];
        float[][] hiddenOutputSig = new float[batch][hidden];

        float[][] finalOutput = new float[batch][output];
        float[][] finalOutputSig = new float[batch][output];       
        
        float[][] hiddenBiasesGrad = new float[batch][hidden];
        float[][] hiddenWeightsGrad = new float[input][hidden];
        
        float[][] finalBiasesGrad = new float[batch][output];
        float[][] finalWeightsGrad = new float[hidden][output];
        
        //Training net
        hiddenOutput = mul(trainData,hiddenWeights,true,true);
        hiddenOutputSig = sigmoid(hiddenOutput,hiddenBiases);
        finalOutput = mul(hiddenOutputSig,finalWeights,true,true);
        finalOutputSig = sigmoid(finalOutput,finalBiases);
        
        //Gradients of weights and biases                
        finalBiasesGrad = hadMul(sub(finalOutputSig,label),dsigmoid(finalOutput,finalBiases));
        finalWeightsGrad = mul(hiddenOutputSig,finalBiasesGrad,false,true);
        hiddenBiasesGrad = hadMul(mul(finalBiasesGrad,finalWeights,true,false),dsigmoid(hiddenOutput,hiddenBiases));
        hiddenWeightsGrad = mul(trainData,hiddenBiasesGrad,false,true);
        
        if(print) {
            //Print gradients
            print(col(hiddenBiasesGrad));
            System.out.println("*************************************");
            print(col(finalBiasesGrad));
            System.out.println("*************************************");
            print(hiddenWeightsGrad);
            System.out.println("*************************************");
            print(finalWeightsGrad);
            System.out.println("*************************************");
        }
        //Return cost
        return cost(finalOutputSig,label);
    }
    
    public static float[][] eps(float[][] a, int a1, int a2, float epsilon) {
        float[][] x = clone(a);
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                x[i][j] = a[i][j];
            }
        }
        x[a1][a2] = a[a1][a2] + epsilon;
        return x;
    }
    
    public static float[][] clone(float[][] a) {
        float[][] x = new float[a.length][a[0].length];
        return x;
    }
    
    public static void update(float a[][], float b[][], float l) {
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                a[i][j] -= l*b[i][j];
            }
        }
    }
    
    //http://www.i-programmer.info/programming/theory/2744-how-not-to-shuffle-the-kunth-fisher-yates-algorithm.html
    public static void sample(float lab[][], float labSamp[][], float train[][], float trainSamp[][], int b)
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
    
    public static float[][] mul(float[][] a, float[][] b, boolean left, boolean right) {
        int m1,n1;
        int m2,n2;
        
        float[][] c;
        
        if(left) {
            if(right) {
                m1 = a.length;
                n1 = a[0].length;
                m2 = b.length;
                n2 = b[0].length;
                
                c = new float[m1][n2];
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
                
                c = new float[m1][n2];
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
                
                c = new float[m1][n2];
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
                
                c = new float[m1][n2];
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
    
    public static void rand(float a[][])
    {
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                Random r = new Random();
                a[i][j] = (float)r.nextGaussian();
            }
        }
    }
    
    public static float[][] sigmoid(float a[][], float b[][])
    {
        float[][] x = new float[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = (float)(Math.exp(a[i][j])*Math.exp(b[0][j])/(Math.exp(a[i][j])*Math.exp(b[0][j])+1));
            }
        }
        
        return x;
    }
    
    public static float[][] dsigmoid(float a[][], float b[][])
    {
        float[][] x = new float[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = (float)(Math.exp(a[i][j])*Math.exp(b[0][j])/((Math.exp(a[i][j])*Math.exp(b[0][j])+1)*(Math.exp(a[i][j])*Math.exp(b[0][j])+1)));
            }
        }
        
        return x;
    }

    public static float cost(float a[][], float b[][])
    {
        float x = 0;
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x+= (a[i][j] - b[i][j])*(a[i][j] - b[i][j]);
            }
        }
        
        x/= 2;
        return x;
    }

    public static float[][] hadMul(float a[][], float b[][])
    {
        float[][] x = new float[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]*b[i][j];
            }
        }
        
        return x;
    }
    
    public static float[][] sub(float a[][], float b[][])
    {
        float[][] x = new float[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[i][j] = a[i][j]-b[i][j];;
            }
        }
        
        return x;
    }

    public static float[][] col(float a[][])
    {
        float[][] x = new float[1][a[0].length];
        
        for(int i=0; i<a.length; i++){
            for(int j=0; j<a[0].length; j++){
                x[0][j]+=a[i][j];
            }
        }
        
        return x;
    }
    
    public static void print(float[][] a) {        
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                System.out.printf("%.4f ",a[i][j]);
            }
            System.out.println();
        }
        System.out.println();
    }
        
    public static void read(float a[][], String b) throws FileNotFoundException
    {
        Scanner scanner = new Scanner(new File(b));
        scanner.useDelimiter(","); 
        
        int i=0;
        int j=0;
        while(scanner.hasNext()){
            String current = scanner.next();
     
            if(current.contains("\n")){
                int index = current.indexOf("\n");
                String first = current.substring(0,index);
                String second = current.substring(index+1);

                a[i][j] = Float.valueOf(first);
                j=0;
                i++;

                a[i][j] = Float.valueOf(second);
                j++;
            }
            
            else{
                a[i][j] = Float.valueOf(current);
                j++;
            }            
        }
        
        scanner.close();
    }

    public static void write(float a[][], String b) throws FileNotFoundException
    {
        PrintWriter out = new PrintWriter(b);
        
        for (int i=0; i<a.length; i++) 
        {
            for (int j=0; j<a[0].length; j++) 
            {
               if((j+1)<a[0].length)
               {
                   out.print(a[i][j] + ",");
               }
               else if(!((j+1)<a[0].length)&&(i+1)<a.length)
               {
                   out.println(a[i][j]);
               }
               
               else
               {
                   out.print(a[i][j]);
               }
            }
        }
         
        out.close();
    }
    
    public static float[][] max(float a[][])
    {
        float[] max = new float[a.length];
        float[][] x = new float[a.length][a[0].length];
        
        for(int i=0; i<a.length; i++)
        {
            float m=0;
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
    
    public static float accuracy(float a[][], float b[][])
    {
        float numerator = 0;
        float denominator = a.length;
        
        for(int i=0; i<a.length; i++)
        {
            float flag = 0;
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