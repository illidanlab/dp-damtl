/* Files end with "_one" can Run on one computer using shell script
 * The port, index, latency parameters are read from outside input
 * 
 * This file is to test synchronous mechanism
 * */

import java.io.*;
import java.net.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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


public class Server_ProxTrace_one_sync {
	
	private static int Blas = 0;
	private static int index; // index we received from specific task node
	private static double value_all = 0.0; // overall objective function value

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
				//X_load = MatrixIO.loadCSV("AlzheimerCSVdatac"+Integer.toString(i+1));
				X_load = MatrixIO.loadCSV("AlzheimerCSVdatar"+Integer.toString(i+1));
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
				//y_load = MatrixIO.loadCSV("labelc"+Integer.toString(i+1));
				y_load = MatrixIO.loadCSV("label"+Integer.toString(i+1));
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
				//y_load = MatrixIO.loadCSV("AlzheimerCSVlabel"+Integer.toString(i+1));
				y_load = MatrixIO.loadCSV("AlzheimerCSVscore"+Integer.toString(i+1));
			}catch (IOException e2){
				throw new RuntimeException(e2);
			}
			
			y_load = zscore(y_load); // response need to z-score
			DenseMatrix64F y_load_train = new DenseMatrix64F(CommonOps.extract(y_load, 0, row_train[i], 0, 1));
			Y[i] = new AMTL_Matrix(y_load_train,Blas);
			/******************Response read end******************/
		}
		
		// Dimension of the model vector
		System.out.println("Central server side, dimension of the model vector is " + Integer.toString(dim));
		System.out.println("Central server side, number of column of P is " + Integer.toString(P.NumColumns));
		System.out.println("Central server side, number of column of S is " + Integer.toString(S.NumColumns));
		
		// Parameters (change)
		double alpha2 = 1; // Step size in central server side
		System.out.println("Central server side, step size is " +  Double.toString(alpha2));

		double mu = 0.0009; // regularization parameter in proximal mapping in central server side (change)
		System.out.println("Central server side, regularization parameter in proximal mapping is " +  Double.toString(mu));
		
		ArrayList<Integer> tnlist = new ArrayList<Integer>(); // Remember the index of the task node that haven't send gradient to central server 
		for(int i = 0; i<T; i++){ 
			tnlist.add(i);
		}
		ArrayList<ClientMessage_one> clientMsglist = new ArrayList<ClientMessage_one>(); // for each task node, we have one message for it
		ArrayList<Socket> socketlist = new ArrayList<Socket>(); // one socket for each task
		ArrayList<Double> obj_value_list = new ArrayList(); // ArrayList that store overall objective function value

		try {
			
			// Creating a socket by binding the port number. Server is ready to listen 
			// from this port.
			int serverPort = 3457; // (change)
			ServerSocket serverSocket = new ServerSocket(serverPort);
			
			System.out.println("****** Get Ready (Starts listening) ******");
			
			int count = 10000; // we can break while loop after iteration finish, count = iterations

			while(true){ 
				Socket clientSocket = serverSocket.accept(); 
				socketlist.add(clientSocket); // a new socket (task)
				ObjectInputStream ois = new ObjectInputStream(clientSocket.getInputStream( ));
				clientMsglist.add((ClientMessage_one)ois.readObject( )); // ClientMessage_one contains the info of where is this message from, need corresponding socket to send, socket is like the bridge or link between TN and CS
				index = (clientMsglist.get(clientMsglist.size()-1)).getId(); // get the last message's index, in each loop, one and only one grad is received
				tnlist.remove(new Integer(index)); // remove by value, not index
				AMTL_Matrix grad = new AMTL_Matrix((clientMsglist.get(clientMsglist.size()-1)).getVec()); // gradient vector
				
				if(index == -1){
					(clientMsglist.get(clientMsglist.size()-1)).setError(1);
				} else if((clientMsglist.get(clientMsglist.size()-1)).getVec().NumRows == dim){
					// New q vector carried by clientMsg.
					AMTL_Matrix nq = new AMTL_Matrix((clientMsglist.get(clientMsglist.size()-1)).getQnew()); 
					
					for(int i = 0; i<dim; i++){
						Q.setDouble(i, index, nq.getDouble(i,0)); 
					}
										
					// Change the corresponding column of the gradient matrix S with the new gradient vector sent from task node, make sure use index 
					for(int i = 0; i<dim; i++){
						S.setDouble(i, index, grad.getDouble(i,0)); // update S 
					}
				} else{
					System.out.println("The vector of client" + index + "does not match the row number of the matrix!\n Permission Denied\n");
					(clientMsglist.get(clientMsglist.size()-1)).setError(3);
				}

				if (tnlist.isEmpty()){ // grads from all tasks are received, process on them is then performed, 
					// the central server is now blocked from outside world
					// Operations need to be done by server.
					AMTL_Matrix temp = new AMTL_Matrix(S); // store \P^{(k-1)}-\alpha_2 \S^{(k)}
					MatrixOps.Scale(temp, -alpha2);
					MatrixOps.ADD(P, temp, temp);
					Operators backward = new Operators(alpha2);
					System.out.println("Rank of T is: " + MatrixOps.getRank(temp));
					AMTL_Matrix P_new = backward.Prox_Trace(temp, mu);
					System.out.println("Rank of P_new is: " + MatrixOps.getRank(P_new));
					
					for(int i = 0; i<P_new.NumRows; i++){
						for(int j = 0; j<P_new.NumColumns; j++){
							P.setDouble(i, j, P_new.getDouble(i,j)); // update P 
						}
					}
					
					/* here the central server need compensation because of the computation of overall objective function value, 
					 in asynchronous case, the computation is done in each thread but in synchronous, only one time is done, 
					 so we need to stop for a while to the offset time delay (asynchronous case) due to the computation 
					 the latency value here is set to be (T-1) * (computation time of one time) */
					TimeUnit.MILLISECONDS.sleep(10);
					
					for(int i = 0; i<T; i++){ // done before compute overall objective function value, save some time
						for(int j = 0; j<dim; j++){
							grad.setDouble(j,0, P.getDouble(j,index)); // extract that column and send back to corresponding task
						}
						ObjectOutputStream oos = new ObjectOutputStream(socketlist.get(i).getOutputStream());
						clientMsglist.get(i).copyVec(grad);
						oos.writeObject(clientMsglist.get(i));
						oos.flush();
						socketlist.get(i).close();
					}
					
					for(int i = 0; i<T; i++){ // renew 
						tnlist.add(i);
					}
					clientMsglist.clear(); // remove all elements
					socketlist.clear();
					
					/*
					// test begin, classification 
					// compute overall objective function value and store it, logistic loss
					AMTL_Matrix W = new AMTL_Matrix(P.NumRows, P.NumColumns, P.BlasID); // newest model matrix
					MatrixOps.ADD(Q, P, W); // W = P + Q
					for(int i = 0; i<W.NumColumns; i++){ // for each TN
						// objective is logistic loss with l2 regularization here
						double value_each = 0.0; // objective function value of each task
						
						int[] rows = new int[]{0}; 
						int[] columns = new int[dim];
						for(int k = 0; k<dim; k++){
							columns[k] = k;
						}
						for(int j = 0; j<row_train[i]; j++){
							rows[0] = j;
							AMTL_Matrix obj_result = new AMTL_Matrix(1,1,Blas); // x_{i}^{T}*w
							double product = 0.0; // the value of obj_result
							AMTL_Matrix w = new AMTL_Matrix(CommonOps.extract((DenseMatrix64F) W.M, 0, dim, i, i+1), 0);
							AMTL_Matrix x = new AMTL_Matrix(XX[i].getSubMatrix(rows, columns)); // copy a new one to Transpose
							MatrixOps.Transpose(x);
							MatrixOps.MULT(x, w, obj_result);
							product = obj_result.getDouble(0, 0); // value of x_{i}^{T}*w
							value_each = value_each + Math.log( 1 + Math.exp(-1 * YY[i].getDouble(j, 0) * product) ); // log ( 1+e^(-y_{i}*x_{i}^{T}*w) ), log is e based					
						}
						
						value_each = value_each/row_train[i];
						value_all = value_all + value_each;
					}
					value_all = value_all + mu*Norms.Trace_Norm(W); // add unclear norm at the end
					// test end, classification
					*/
					
					// test begin, regression 
					// compute overall objective function value and store it, least square loss
					AMTL_Matrix W = new AMTL_Matrix(P.NumRows, P.NumColumns, P.BlasID); // newest model matrix
					MatrixOps.ADD(Q, P, W); // W = P + Q
					for(int i = 0; i<T; i++){ // for each TN
						double value_each = 0.0; // objective function value of each task
						int[] rows = new int[]{0}; 
						int[] columns = new int[dim];
						for(int k = 0; k<dim; k++){
							columns[k] = k;
						}
						for(int j = 0; j<row_train[i]; j++){
							rows[0] = j;
							AMTL_Matrix obj_result = new AMTL_Matrix(1,1,Blas); // x_{i}^{T}*w
							double product = 0.0; // the value of obj_result
							AMTL_Matrix w = new AMTL_Matrix(CommonOps.extract((DenseMatrix64F) W.M, 0, dim, i, i+1), 0);
							AMTL_Matrix x = new AMTL_Matrix(X[i].getSubMatrix(rows, columns)); // copy a new one to Transpose
							MatrixOps.Transpose(x);
							MatrixOps.MULT(x, w, obj_result);
							product = obj_result.getDouble(0, 0); // value of x_{i}^{T}*w
							value_each = value_each + Math.pow(product-Y[i].getDouble(j, 0),2); 				
						}	
						
						value_each = value_each/row_train[i];
						value_all = value_all + value_each;
					}
					System.out.println(Norms.Trace_Norm(W));
					value_all = value_all + mu*Norms.Trace_Norm(W); // add unclear norm at the end
					// test end, regression
					
					obj_value_list.add(value_all);
					value_all = 0.0; // must do, otherwise will accumulate iteration after iteration
					
					// test begin
					count = count - 1;
					if (count == 0){
						break;
					}
					// test end
				}
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
