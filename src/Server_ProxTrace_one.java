/* Files end with "_one" can Run on one computer using shell script
 * The port, index, latency parameters are read from outside input
 * */

import java.io.*;
import java.net.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.FutureTask;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigInteger;
import org.ejml.ops.CommonOps;
import org.ejml.ops.NormOps;
//import java.security.MessageDigest;
import org.ejml.ops.MatrixIO;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.MatrixOps;
import org.AMTL_Matrix.Norms.Norms;


import org.ejml.data.DenseMatrix64F;


public class Server_ProxTrace_one {
	
	private static int Blas = 0;

	public static DenseMatrix64F zscore(DenseMatrix64F res){
		// perform z-score on whole response
		int row_res = res.numRows;
		DenseMatrix64F res_copy = new DenseMatrix64F(res);
		double mean = CommonOps.elementSum(res_copy)/row_res; // mean of all response
		double sum = 0;
		for(int i=0; i<row_res; i++){
			sum = sum + Math.pow(res_copy.get(i, 0)-mean, 2);
		}
		double std = Math.sqrt(sum/row_res); // standard deviation of all response
		for(int i=0; i<row_res; i++){ // perform z-score on training set
			res_copy.set(i, 0, (res_copy.get(i, 0) - mean)/std);
		}
		
		return res_copy;
	}
	
	public static DenseMatrix64F zscoreData(DenseMatrix64F data){
		// perform z-score on whole data set
		DenseMatrix64F zscored_data = new DenseMatrix64F(data);
		int row = data.numRows;
		for(int i=0; i<data.numCols; i++){ // for each column in data matrix (each feature)
			CommonOps.insert(zscore(CommonOps.extract(data, 0, row, i, i+1)), zscored_data, 0, i);
		}
		
		return zscored_data;
	}
	
	public static DenseMatrix64F normalization(DenseMatrix64F dataset){
		// data normalization, find the largest norm among the data set and divide every data point by this norm, hence project them into a unit ball 
		double norm = 0.0; // norm of a data point
		for(int i=0; i<dataset.numRows; i++){
			DenseMatrix64F D = CommonOps.extract(dataset, i, i+1, 0, dataset.numCols);
			if(NormOps.normF(D) > norm){
				norm = NormOps.normF(D);
			}
        }
		CommonOps.divide(norm, dataset);
		return dataset;
	}
	
	public static void main(String[] args){
		
		/* BlasID 0: ejml
		 * BlasID 1: ujmp
		 * BlasID 2: jama
		 * BlasID 3: jblas
		*/
		
		/***************************************Matrix Initialization start***************************************/
        //Initialization independent components of the model matrix (change)
  		DenseMatrix64F Q_load;
  		try{
  			Q_load = MatrixIO.loadCSV("startP"); // same shape as P_load, so use same file to Initialize
  			System.out.println("Central server side, initial independent component Q is loaded!");
  		}catch (IOException e1){
  			throw new RuntimeException(e1);
  		}

  		AMTL_Matrix Q = new AMTL_Matrix(Q_load,Blas);
      	
		//Initialization shared components of the model matrix (change)
		DenseMatrix64F P_load;
		try{
			P_load = MatrixIO.loadCSV("startP");
			System.out.println("Central server side, initial shared component P is loaded!");
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}

		AMTL_Matrix P = new AMTL_Matrix(P_load,Blas);
		int dim = P.NumRows; // data dimension
		int T = P.NumColumns; // number of TN

		//Initialization S matrix, the matrix contains the gradient w.r.t. p  (change)
		DenseMatrix64F S_load;
		try{
			S_load = MatrixIO.loadCSV("startS");
			System.out.println("Central server side, initial gradient matrix S is loaded!");
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		AMTL_Matrix S = new AMTL_Matrix(S_load,Blas);
		/***************************************Matrix Initialization end***************************************/
		
		ArrayList<Double> obj_value_list = new ArrayList(); // ArrayList that store overall objective function value
		
		// Load training data and label set
		double p_train=0.3;
		AMTL_Matrix[] X = new AMTL_Matrix[T]; // array of training data set
		AMTL_Matrix[] Y = new AMTL_Matrix[T]; // array of training label set
		int[] row_train = new int[T]; // number of training data in each data set
		for(int i = 0; i<T; i++){ // for each TN
			/******************Data read start******************/
			DenseMatrix64F X_load;
			/*
			// Synthetic
			try{ // data matrix X
				X_load = MatrixIO.loadCSV("data"+Integer.toString(i+1)); 
			}catch (IOException e1){
				throw new RuntimeException(e1);
			}
			
			// school.mat
			try{ 
				X_load = MatrixIO.loadCSV("schoolCSVdata"+Integer.toString(i+1));
			}catch (IOException e1){
				throw new RuntimeException(e1);
			}
			X_load = normalization(X_load); // need to normalize
			*/
			// Alzheimer disease
			try{ 
				X_load = MatrixIO.loadCSV("AlzheimerCSVdatac"+Integer.toString(i+1));
				//X_load = MatrixIO.loadCSV("AlzheimerCSVdatar"+Integer.toString(i+1));
			}catch (IOException e1){
				throw new RuntimeException(e1);
			}
			X_load = zscoreData(X_load); // data need to z-score
			X_load = normalization(X_load); // need to normalize
			
			row_train[i] = (int) Math.round(p_train*X_load.numRows); 
			DenseMatrix64F X_load_train = new DenseMatrix64F(CommonOps.extract(X_load, 0, row_train[i], 0, dim));
			X[i] = new AMTL_Matrix(X_load_train,Blas);	
			/******************Data read end******************/
			
			/******************Response read start******************/
			DenseMatrix64F y_load;
			/*
			// Synthetic
			try{ // label vector y
				y_load = MatrixIO.loadCSV("labelc"+Integer.toString(i+1));
				//y_load = MatrixIO.loadCSV("label"+Integer.toString(i+1));
			}catch (IOException e2){
				throw new RuntimeException(e2);
			}
			
			// school.mat
			try{ 
				y_load = MatrixIO.loadCSV("schoolCSVscore"+Integer.toString(i+1));
			}catch (IOException e2){
				throw new RuntimeException(e2);
			}
			*/
			// Alzheimer disease 
			try{ // label vector y
				y_load = MatrixIO.loadCSV("AlzheimerCSVlabel"+Integer.toString(i+1));
				//y_load = MatrixIO.loadCSV("AlzheimerCSVscore"+Integer.toString(i+1));
			}catch (IOException e2){
				throw new RuntimeException(e2);
			}
			
			//y_load = zscore(y_load); // response need to z-score
			DenseMatrix64F y_load_train = new DenseMatrix64F(CommonOps.extract(y_load, 0, row_train[i], 0, 1));
			Y[i] = new AMTL_Matrix(y_load_train,Blas);
			/******************Response read end******************/
		}
		
		// Dimension of the model vector
		System.out.println("Central server side, dimension of the model vector is " + Integer.toString(dim));
		System.out.println("Central server side, number of column of P is " + Integer.toString(P.NumColumns));
		System.out.println("Central server side, number of column of S is " + Integer.toString(S.NumColumns));
		
		// Parameters (change)
		double alpha2 = 0.1; // Step size in central server side
		System.out.println("Central server side, step size is " +  Double.toString(alpha2));

		double mu = 0.002; // regularization parameter in proximal mapping in central server side (change)
		System.out.println("Central server side, regularization parameter in proximal mapping is " +  Double.toString(mu));
		
		
		try {
			
			// Creating a socket by binding the port number. Server is ready to listen 
			// from this port.
			int serverPort = 3457; // (change)
			ServerSocket serverSocket = new ServerSocket(serverPort);
			
			System.out.println("****** Get Ready (Starts listening) ******");
			
			int count = 200000; // we can break while loop after totally iteration finish, count = sum of iterations
			
			while(true){
				
				// accept(): A blocking method call. When 1 client contacts, the method is unblocked 
				// and returns a Socket object to the server to communicate with the client. 
				Socket clientSocket = serverSocket.accept();
				System.out.println("Starts communicating a client.");
				
				ServerThread_Trace_one t = new ServerThread_Trace_one(clientSocket, X, Y, row_train, dim, Q, P, S, alpha2, mu);
				// FutureTask interface takes a callable object. Object ft is used to call the call() 
				// method overridden in ServerThread.
				FutureTask<MultiReturn> ft = new FutureTask<MultiReturn>(t);
				// This invokes the thread where call() method of ServerThread operates.
				new Thread(ft).start();
				// get() is a method of FutureTask and returns the result of 
				// the computations of call() method of ServerThread.
				MultiReturn next = (MultiReturn) ft.get(); // save for next iteration
				Q = next.Q; // should be updated here
				P = next.P;
				S = next.S;
				obj_value_list.add(next.v);
				// test begin
				count = count - 1;
				if (count == 0){
					break;
				}
				// test end
			}
			
			// test begin
			int l = obj_value_list.size();
			DenseMatrix64F a = new DenseMatrix64F(1, l); // ArrayList<Double> to DenseMatrix64F
			for(int i = 0; i<l; i++){
				a.set(0, i, obj_value_list.get(i));
			}
			try{ 
				MatrixIO.saveCSV(a,"objvalue"); // length of this file should be the sum of all iterations of all tasks, in pure asynchronous case
			}catch (IOException e1){
				throw new RuntimeException(e1);
			}
			System.out.println("Write successfully!");
			// test end
			
			
		} catch (Exception e){
			e.printStackTrace();
		}
	}
	
}
