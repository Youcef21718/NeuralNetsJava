import java.util.Random;
import java.lang.Math;

import java.io.File;
import java.io.PrintWriter;
import java.util.Scanner;
import java.io.FileNotFoundException;

public class trainBigStoch2
{
    private static long startTime = System.currentTimeMillis();
    public static void main(String[] args) throws FileNotFoundException {
        //Hyperparameters
        int numEpochs = 30;
        int batch = 10;
        float learningRate = (float)3/batch;
        
        //Training neural net
        int x = 60000;
        int y = 785;
        int numHidden = 30;
        int digits = 10;
        
        float[][] label = new float[x][digits];
        float[][] trainData = new float[x][y-1];
        
        float[][] labelSample = new float[batch][digits];
        float[][] trainDataSample = new float[batch][y-1];
                                
        float[][] hiddenWeights = new float[y-1][numHidden];
        float[][] hiddenWeightsGrad = new float[y-1][numHidden];
                
        float[][] hiddenBiases = new float[1][numHidden];
        float[][] hiddenBiasesGrad = new float[batch][numHidden];
        
        float[][] hiddenOutput = new float[batch][numHidden];
        float[][] hiddenOutputSig = new float[batch][numHidden];
        
        float[][] finalWeights = new float[numHidden][digits];
        float[][] finalWeightsGrad = new float[numHidden][digits];
        
        float[][] finalBiases = new float[1][digits];
        float[][] finalBiasesGrad = new float[batch][digits];
        
        float[][] finalOutput = new float[batch][digits];
        float[][] finalOutputSig = new float[batch][digits];
        
        //Testing neural net
        int xTest = 10000;
        int yTest = 785;
        
        float[][] labelTest = new float[xTest][digits];
        float[][] testData = new float[xTest][yTest-1];
                
        //Initialize labels and data
        read(trainData,"data/trainData.csv");
        read(label,"data/label.csv");
        read(testData,"data/testData.csv");
        read(labelTest,"data/labelTest.csv");
        
        //Initialize weights and biases
        rand(finalWeights);
        rand(hiddenWeights);
        rand(finalBiases);
        rand(hiddenBiases);
        
        int epochTime = x/batch;
        for(int i=0; i<numEpochs; i++){
            for(int j=0; j<epochTime; j++){
                //Select random batch
                sample(label, labelSample, trainData, trainDataSample, batch);
                
                //Training net
                hiddenOutput = mul(trainDataSample,hiddenWeights,true,true);
                hiddenOutputSig = sigmoid(hiddenOutput,hiddenBiases);
                finalOutput = mul(hiddenOutputSig,finalWeights,true,true);
                finalOutputSig = sigmoid(finalOutput,finalBiases);
                
                //Gradients of weights and biases                
                finalBiasesGrad = hadMul(sub(finalOutputSig,labelSample),dsigmoid(finalOutput,finalBiases));
                finalWeightsGrad = mul(hiddenOutputSig,finalBiasesGrad,false,true);
                hiddenBiasesGrad = hadMul(mul(finalBiasesGrad,finalWeights,true,false),dsigmoid(hiddenOutput,hiddenBiases));
                hiddenWeightsGrad = mul(trainDataSample,hiddenBiasesGrad,false,true);
                
                //Update weights and biases
                update(finalBiases,finalBiasesGrad,learningRate);
                update(hiddenBiases,hiddenBiasesGrad,learningRate);
                update(finalWeights,finalWeightsGrad,learningRate);
                update(hiddenWeights,hiddenWeightsGrad,learningRate);
                
                //Print results
                //System.out.println(cost(finalOutputSig,labelSample));
                //System.out.println(accuracy(labelSample,max(finalOutputSig)));
                
                //Display program state                
                if((i*epochTime+j)%100==0)
                {
                    System.out.println(i*epochTime+j);
                }
           }
        }
        
        //Training network
        System.out.print(
        cost(sigmoid(mul(sigmoid(mul(trainData,hiddenWeights,true,true),hiddenBiases),finalWeights,true,true),finalBiases),label)+","+           
        accuracy(label,max(sigmoid(mul(sigmoid(mul(trainData,hiddenWeights,true,true),hiddenBiases),finalWeights,true,true),finalBiases)))+","   
        );
            
        //Test network
        System.out.println(   
        cost(sigmoid(mul(sigmoid(mul(testData,hiddenWeights,true,true),hiddenBiases),finalWeights,true,true),finalBiases),labelTest)+","+           
        accuracy(labelTest,max(sigmoid(mul(sigmoid(mul(testData,hiddenWeights,true,true),hiddenBiases),finalWeights,true,true),finalBiases)))   
        );

        //Save training session
        //write(finalWeights,"finalWeights.txt");
        //write(hiddenWeights,"hiddenWeights.txt");
        //write(finalBiases,"finalBiases.txt");
        //write(hiddenBiases,"hiddenBiases.txt");
        
        //Display program execution time
        long endTime = System.currentTimeMillis();
        System.out.println("It took " + (endTime - startTime) + " milliseconds");
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