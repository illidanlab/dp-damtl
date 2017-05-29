/*Model matrix generation
 *First generate a matrix A\in R^{T\times r} from N(0,0.5), where T is the number of task and r is smaller than T. 
 *Then generate B\in R^{r\times d} from N(0,0.5), where d is the data dimension. Then set P = A*B, where P is the shared part. 
 *Then generate Q\in R^{T\times d} from N(0,0.01), Q is the independent part. 
 *Finally set W=P+Q, where W\in R^{T\times d} is the model matrix. P is low rank, Q is full rank, W is full rank. 
 */
import java.util.*;
import java.io.*;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixFeatures;
import org.ejml.ops.MatrixIO;

public class ModelMatrix {

	/**
	 * @param args
	 */
	private int T; // # of task, also in Initialization.py file
	private int d; // dimension of data
	private int r = 3; // number of column of matrix low1, also is the rank of P
	private DenseMatrix64F P;  // an T*d matrix to store the independent part of task model, one row for one model
	private DenseMatrix64F Q;  // an T*d matrix to store the shared part of task model, one row for one model
	private DenseMatrix64F W;  // an T*d matrix to store the task model, one row for one model
	
	ModelMatrix(int T, int d){
		System.out.println("Model matrix created!");
		this.T = T;
		this.d = d;
	}
	
	public void generateP() {
		// Generate independent part of task model matrix
		
		// Generate a low rank matrix A: T*r as left multiplier, which control the rank of P as r, with each element generated from N(0,0.5), N for normal distribution 
		double[][] low1 = new double[T][r]; 
		for(int i=0; i<=T-1; i++){
			for(int j=0; j<=r-1; j++){
				Random randomno = new Random();
				low1[i][j] = randomno.nextGaussian()/2.0; // i+j;//
			}
		}
		
		// Generate a low rank matrix B: r*d as right multiplier, each column of it contains the weights, corresponding to columns of low1, with each element generated from N(0,0.5) 
		double[][] low2 = new double[r][d]; 
		for(int i=0; i<=r-1; i++){
			for(int j=0; j<=d-1; j++){
				Random randomno = new Random();
				low2[i][j] = randomno.nextGaussian()/2.0; // i+j;//
			}
		}

		//Compute matrix P using outer product of low1*low2
		DenseMatrix64F res = new DenseMatrix64F(T, d); //matrix product of low1 and low2
		CommonOps.mult(new DenseMatrix64F(low1), new DenseMatrix64F(low2), res);
		//System.out.println(MatrixFeatures.rank(res)); // test the rank 
		this.P = res;
		}
	
	public void generateQ() {
		// Generate shared part of task model matrix, a matrix with each element generated from N(0,0.01)
		double[][] res = new double[T][d]; 
		for(int i=0; i<=T-1; i++){
			for(int j=0; j<=d-1; j++){
				Random randomno = new Random();
				res[i][j] = randomno.nextGaussian()/100.0; // 1;//
			}
		}
		this.Q = new DenseMatrix64F(res); 
	}
	
	public void generateW() {
		//Generate matrix W = P + Q
		DenseMatrix64F m = new DenseMatrix64F(T, d);
		CommonOps.add(this.P, this.Q, m); // don't use "+", use CommonOps.add
		this.W = m;
	}
	
	public void writefile(){
		// write the task model matrix into a file
		try{
			MatrixIO.saveCSV(new DenseMatrix64F(W),"taskmodels"); // the file is stored under project/tm1
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}	
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		ModelMatrix a = new ModelMatrix(20,28);
		a.generateP();
		a.generateQ();
		a.generateW();
		a.writefile();
		
	}
	
}
