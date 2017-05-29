/* Synthetic data generation
 * every element of each data point is draw from N(0,0.01), then normlize them so that all data points are in a unit ball. 
 * Then use number t task model generated from last step, use dot product, 
 * to generate the response of each data point in this data set for regression, 
 * further use 0 as boundary to transfer these response to +1 or -1 for classification.
 */

import java.util.*;
import java.io.*;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
import org.ejml.ops.MatrixIO;
import org.AMTL_Matrix.*;

public class Syntheticdata { // one instance for one data set
	
	private int nt; // # of data in this task
	private int p=0; // # of data with positive label in this task, +1 for classification
	private DenseMatrix64F data;  // an nt*d matrix to store the data, one row for one data
	private DenseMatrix64F label;  // an d dimensional vector to store the label
	private DenseMatrix64F labelc;  // an d dimensional vector to store the label of classification
	private static int d=28; // dimension of data
	private static int T=20; // # of task
	private static int count = 0; // count number of data set already created
	
	Syntheticdata(int nt){
		System.out.println("Synthetic data created!");
		++count;
		System.out.println("Totally " + count + " data set(s) created");
		this.nt = nt;
	}
	
	public static DenseMatrix64F normalization(DenseMatrix64F dataset){
		// data normalization, find the largest norm among the data set and divide every data point by this norm, hence project them into a unit ball 
		double norm = 0.0; // norm of a data point
		for(int i=0; i<dataset.numRows; i++){
			DenseMatrix64F D = CommonOps.extract(dataset, i, i+1, 0, d);
			if(NormOps.normF(D) > norm){
				norm = NormOps.normF(D);
			}
        }
		CommonOps.divide(norm, dataset);
		return dataset;
	}
	
	public void generatedata() {
		//Generate data matrix with each element generated from N(0,0.01) 
		double[][] tmp1 = new double[nt][d]; 
		for(int i=0; i<=nt-1; i++){
			for(int j=0; j<=d-1; j++){
				Random randomno = new Random();
				tmp1[i][j] = randomno.nextGaussian(); // i+j;//
			}
		}
		DenseMatrix64F tmp2 = new DenseMatrix64F(tmp1);
		data = normalization(tmp2); // data normalization  
	}
	
	public void generatelabel(DenseMatrix64F w) {
		//Generate label vector 
		DenseMatrix64F tmp = new DenseMatrix64F(nt,1); // label vector, column
		DenseMatrix64F wtrans = new DenseMatrix64F(d,1); // temporarily store the one transposed task model w
		CommonOps.mult(data, CommonOps.transpose(w, wtrans), tmp);
		label = tmp;
	}
	
	public void labelTrans(){
		// positive or zero label to +1, negative label to -1
		labelc = label.copy();
		for(int i=0; i<=nt-1; i++){
			if(label.get(i, 0)<0){
				labelc.set(i, 0, -1);
			}else{
				labelc.set(i, 0, +1);
				this.p++; // positive label increase one
			}
		}
	}

	public void info(){
		// Return some information 
		System.out.println("Totally " + this.p + " data with positive label");
	}
	
	public void writefile(){
		// write the data and label into a file
		try{
			MatrixIO.saveCSV(data,"data"+Integer.toString(count)); // the file is stored under project/data(count+1)
			MatrixIO.saveCSV(label,"label"+Integer.toString(count)); // the file is stored under project/label(count+1)
			MatrixIO.saveCSV(labelc,"labelc"+Integer.toString(count)); // the file is stored under project/labelc(count+1)
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		DenseMatrix64F taskModel;
		
		try{ 
			taskModel = MatrixIO.loadCSV("taskmodels"); // load task model matrix
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		
		Syntheticdata[] dataSetArray = new Syntheticdata[T]; // an array of synthetic data sets
		for ( int i=0; i<dataSetArray.length; i++) {
			dataSetArray[i]=new Syntheticdata(200);
			dataSetArray[i].generatedata(); 
			dataSetArray[i].generatelabel(CommonOps.extract(taskModel, i, i+1, 0, d)); // extract one row
			dataSetArray[i].labelTrans();
			dataSetArray[i].writefile();
			//dataSetArray[i].info();
		}
	}
}