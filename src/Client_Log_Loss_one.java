import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.TimeUnit;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixIO;
import org.ejml.ops.NormOps;

import java.io.IOException;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.*;
import org.AMTL_Matrix.Norms.Norms;

public class Client_Log_Loss_one {
	// Task node side, use logistic loss in classification problem
	
	private static int serverPort = 3457;
	private static int index; // Which task node we are in
	private static int latency; // wait as time delay in network, in MILLISECONDS, here use int
	private static int countITER = 0; // show the number of current iteration 
	private static double Lambda = Math.pow(10, -3); // regularization parameter, also one in.py file
	private static double eps1 = 0.5; // privacy parameter on q
	private static double eps2 = 70.0; // privacy parameter on grad
	private static double S1 = Math.pow(10, -2)/1.2; // Remember divide by p_train, sensitivity on q_new, Math.pow(10, -3)/1.2 for Synthetic data sets 5, Math.pow(10, -2)/1.2 for Alzheimer
	private static double S2 = Math.pow(10, -1)/1.2; // Remember divide by p_train, sensitivity on output gradient, Math.pow(10, -2)/1.2 for Synthetic data sets 5, Math.pow(10, -1)/1.2 for Alzheimer
	private static ArrayList<String> error = new ArrayList<String>(); // array of error rate in all iterations
	private static int ITER = 10000; // Number of iterations (change)
	private static double alpha1 = 0.1; // gradient descent (change), remember to change S1 when change alpha1
	private static int Blas = 0;
	private static double p_train = 0.3; // percentage of training set, also in Initialization.py file
	
	public static void measurec(AMTL_Matrix data, AMTL_Matrix label, AMTL_Matrix model){
		// Error rate of task model in each iteration, classification problem
		int row = data.NumRows;
		int col = data.NumColumns;
		double err = 0.0; // number of error already made 
		int[] rows = new int[]{0}; // index of row, for data matrix
		int[] columns = new int[col]; // index of columns, for extraction one row from data matrix		
		AMTL_Matrix x = new AMTL_Matrix(1, col, 0); // store one data point, one row of X, hence a row vector
		for(int i = 0; i<col; i++){
			columns[i] = i;
		}
		
		for(int i = 0; i < row; i = i + 1) {
			double sum = 0; // running sum in dot product
			rows[0] = i;
			x = data.getSubMatrix(rows, columns); // one data point, one row of X
			for(int j = 0; j < col; j = j + 1) {
				sum = sum + x.getDouble(j, 0)*model.getDouble(j, 0);
			}
			sum = sum*label.getDouble(i, 0);

			if (sum<0){
				err = err + 1;
			}
		}
		
		String e = Double.toString(err/row);
		if ((countITER % 10) == 0)
		{
			System.out.println("This is task node " + Integer.toString(index) + ", the error rate in iteration " + Integer.toString(countITER) + " is " + e);
		}
		error.add(e); // ArrayList doesn't accept primitive numbers here
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
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//InputStream is;
		//InputStreamReader isr;
		//BufferedReader br;
		
		
		
		// Blas is the ID that tells us which BLAS library we would like to use.
		/* BlasID 0: ejml
		 * BlasID 1: ujmp
		 * BlasID 2: jama
		 * BlasID 3: jblas
		*/
		
		// The index, latency parameters are read from outside input
		index = Integer.parseInt(args[0]);
		latency = Integer.parseInt(args[1]);
		
		System.out.println("Task node index is: " + index);
		System.out.println("Task node latency is: " + latency);
		
		/**************************************************Data read start**************************************************/
		// objective is logistic loss with l2 regularization. See operators. (change) from this line
		DenseMatrix64F X_load;
		/*
		// Synthetic
		try{ // data matrix X
			X_load = MatrixIO.loadCSV("data"+index); // "data"+index works
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		*/
		// Alzheimer disease
		try{ 
			X_load = MatrixIO.loadCSV("AlzheimerCSVdatac"+index);
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		X_load = zscoreData(X_load); // data need to z-score
		X_load = normalization(X_load); // need to normalize
		
		System.out.println("Data set "+ Integer.toString(index) +" is loaded!");
	
		// Testing testing error 
		int row_train = (int) Math.round(p_train*X_load.numRows); // number of rows in training set
		System.out.println("Number of rows in training set: " + Integer.toString(row_train));
		int dim = X_load.numCols; 
		System.out.println("Dimension of the feature vector is " + Integer.toString(dim)); // Dimension of the feature vectors

		DenseMatrix64F X_load_train = new DenseMatrix64F(CommonOps.extract(X_load, 0, row_train, 0, dim)); // round above
		DenseMatrix64F X_load_test = new DenseMatrix64F(CommonOps.extract(X_load, row_train, X_load.numRows, 0, dim));
		AMTL_Matrix X_train = new AMTL_Matrix(X_load_train,Blas); // training data 
		AMTL_Matrix X_test = new AMTL_Matrix(X_load_test,Blas); // testing data
		/**************************************************Data read end**************************************************/
		
		/************************************************Response read start************************************************/
		DenseMatrix64F y_load;
		/*
		// Synthetic
		try{ // label vector y
			y_load = MatrixIO.loadCSV("labelc"+index);
		}catch (IOException e2){
			throw new RuntimeException(e2);
		}
		*/
		// Alzheimer disease 
		try{ // label vector y
			y_load = MatrixIO.loadCSV("AlzheimerCSVlabel"+index);
		}catch (IOException e2){
			throw new RuntimeException(e2);
		}
		
		System.out.println("Label set "+ Integer.toString(index) +" is loaded!");
		
		// Testing testing error 
		DenseMatrix64F y_load_train = new DenseMatrix64F(CommonOps.extract(y_load, 0, row_train, 0, 1)); // initialization, round above
		DenseMatrix64F y_load_test = new DenseMatrix64F(CommonOps.extract(y_load, row_train, X_load.numRows, 0, 1));
		AMTL_Matrix y_train = new AMTL_Matrix(y_load_train,Blas); // training response 
		AMTL_Matrix y_test = new AMTL_Matrix(y_load_test,Blas); // testing response
		/************************************************Response read end************************************************/
		
		/************************************************Initialization start************************************************/
		// Read the initial q, use q_{t}^{0} in task t, do only one time for each task, for whole project 
		DenseMatrix64F q0;
		try{ 
			q0 = CommonOps.extract(MatrixIO.loadCSV("startQ"), index-1, index, 0, dim); // load corresponding row of task model matrix, note that when we use index data set, the task model should enter (index-1 to index), q0 is a row vector
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		CommonOps.transpose(q0); // transform to column vector
		AMTL_Matrix q_ini = new AMTL_Matrix(q0,Blas); // row vector
		System.out.println("Task node "+ Integer.toString(index) +" initial q is loaded!");
				
		AMTL_Matrix p_new; // New p, received from central server 
		AMTL_Matrix q_old = new AMTL_Matrix(q0,Blas); // Old q, computed and left in this task node in last time, note the fact that it is equal to q0 now is just to initialize it to some value, it has nothing to do with q0, it will assign to other values later 
		AMTL_Matrix q_new; // New q 
		AMTL_Matrix grad; // Output gradient 
		
		// Creating an object of ClientMessage_one class. 
		ClientMessage_one clientMsg = new ClientMessage_one(dim,Blas);
		/************************************************Initialization end************************************************/

		// This is a standard way to keep the time.
		Date start_time = null;
		
		try{
			// Set the socket
			InetAddress serverHost = InetAddress.getByName("localhost"); // 127.0.0.1 in address.txt indicates the localhost (lab computer) itself
			
			// Server port number should be same as the number defined in Server_ProxTrace.java
			Socket clientSocket = null;
			
			// Start the timer
			start_time = new Date();			
			
			// Start work
			for(int j = 0; j < ITER; j++){
				countITER = j;
						
				// In every iteration a new socket object is created.  
				// I will check whether we need to create a new object in each iteration
				// or we can create one outside the loop once.
				clientSocket = new Socket(serverHost, serverPort);
				ObjectOutputStream oos;
				ObjectInputStream ois;
				
				// Send a message (the gradient w.r.t p, at old p) to server and this unblock the accept() method and 
				// invokes a communication. 
				// Serializing the vector.
				oos = new ObjectOutputStream(clientSocket.getOutputStream());
				clientMsg.copyId(index-1); // int field in class ClientMessage_one is initialized as 0, need to change
				oos.writeObject(clientMsg);
				TimeUnit.MILLISECONDS.sleep(latency);
				oos.flush();

				// Get the message (new p) at the end of the operation at server's end.
				ois = new ObjectInputStream(clientSocket.getInputStream());
				clientMsg = (ClientMessage_one)ois.readObject( );

				if(clientMsg.getError() == 0){
					// Operation needs to be done at client end.	
						
					p_new = new AMTL_Matrix(clientMsg.getVec()); // New p, better to new a new AMTL_Matrix type
					/*
					// non-DP version start		
					//Forward step 1 (Gradient Update on q)
					Operators forward1 = new Operators(alpha1);

					if(j == 0){
						q_new = forward1.LogLossReg_Forward(X_train, y_train, p_new, q_ini, Lambda); // First iteration q is initialized by the minimizer of L(q) with p=0
					}else{
						q_new = forward1.LogLossReg_Forward(X_train, y_train, p_new, q_old, Lambda); // Other iteration q is the previous one
					}

					// task model performance 
					AMTL_Matrix w = new AMTL_Matrix(p_new.NumRows, p_new.NumColumns, p_new.BlasID);; // w = p_new + q_old, for consistency 
					MatrixOps.ADD(p_new, q_old, w); // w = p + q, pay attention here p_new corresponding to q_old
					measurec(X_test, y_test, w);

					// Store for the use in next iteration, not q_new_n, see algorithm
					q_old = new AMTL_Matrix(q_new); // q_old has its own storage space
					
					// Forward step 2 (Compute output gradient)
					Operators forward2 = new Operators(alpha1); // StepSize alpha1 here is useless since we only use part of LogLossReg_Forward as LogLossGrad_Forward and there is no gradient update involved in it					
					grad = forward2.LogLossGrad_Forward(X_train, y_train, p_new, q_new); // p here should be p_new and q should be q_new
					// Update client message with the updated vector.
					clientMsg.copyVec(grad); // Should send grad_n here
					clientMsg.copyQnew(q_new);
					clientMsg.copyId(index-1);
					// non-DP version end
					*/
					
					// DP version start
					// Generate two noise vectors
					DPNoise dpn1 = new DPNoise(dim, S1, eps1);
					dpn1.compute();
					AMTL_Matrix Dpn1 = new AMTL_Matrix(dpn1.output(), Blas); // Constructor: AMTL_Matrix(Object Input, int BlasID)
					DPNoise dpn2 = new DPNoise(dim, S2, eps2);
					dpn2.compute();
					//AMTL_Matrix Dpn2 = new AMTL_Matrix(dim, 1, Blas); // A test of set noise vector on grad to 0
					AMTL_Matrix Dpn2 = new AMTL_Matrix(dpn2.output(), Blas);
					
					//Forward step 1 (Gradient Update on q)
					Operators forward1 = new Operators(alpha1);

					if(j == 0){
						q_new = forward1.LogLossReg_Forward(X_train, y_train, p_new, q_ini, Lambda); // First iteration q is initialized by the minimizer of L(q) with p=0
					}else{
						q_new = forward1.LogLossReg_Forward(X_train, y_train, p_new, q_old, Lambda); // Other iteration q is the previous one
					}
					
					// Add noise on q_new
					AMTL_Matrix q_new_n = new AMTL_Matrix(q_new.NumRows, q_new.NumColumns, q_new.BlasID);
					MatrixOps.ADD(Dpn1, q_new, q_new_n);

					// task model performance 
					AMTL_Matrix w = new AMTL_Matrix(p_new.NumRows, p_new.NumColumns, p_new.BlasID);; // w = p_new + q_old, for consistency 
					MatrixOps.ADD(p_new, q_old, w); // w = p + q, pay attention here p_new corresponding to q_old
					measurec(X_test, y_test, w); // Use testing set here! err when w = q_ini: if index == 1: err = 0.185, if index == 2: err = 0.225
					
					// Store for the use in next iteration, not q_new_n, see algorithm
					q_old = new AMTL_Matrix(q_new); // q_old has its own storage space
					
					// Forward step 2 (Compute output gradient)
					Operators forward2 = new Operators(alpha1); // StepSize alpha1 here is useless since we only use part of LogLossReg_Forward as LogLossGrad_Forward and there is no gradient update involved in it					
					grad = forward2.LogLossGrad_Forward(X_train, y_train, p_new, q_new_n); // p here should be p_new and q should be q_new_n
					
					// Add noise on grad
					AMTL_Matrix grad_n = new AMTL_Matrix(grad.NumRows, grad.NumColumns, grad.BlasID);
					MatrixOps.ADD(Dpn2, grad, grad_n);
					
				    // Update client message with the updated vector.
					clientMsg.copyVec(grad_n); // Should send grad_n here
					clientMsg.copyQnew(q_new);
					clientMsg.copyId(index-1);
					// DP version end
					
				
					} else if(clientMsg.getError() == 1){
					System.out.println("Error Message 1: Permission Denied! ");
					System.exit(0);
				    } else if(clientMsg.getError() == 3){
					System.out.println("Error Message 3: Vector Length Error!\nPlease change the ROW variable and restart the program!\nExit");
					System.exit(0);
				}else{
					// You can set more kinds of error in the server
					System.out.println("Unknown Error!");
				}
				oos.close();
				ois.close();
			}
			
			DenseMatrix64F a = new DenseMatrix64F(1, error.size()); // in matrix form
			for(int i=0; i<error.size(); i++){
	            a.set(0, i,  Double.parseDouble(error.get(i)));
			} 
			try{ // write at the end of program
				MatrixIO.saveCSV(a,"errorRate"+Integer.toString(index)); // the file is stored under /Javaws/DAMTLDP
			}catch (IOException e1){
				throw new RuntimeException(e1);
			}
			System.out.println("Write successfully!");
			
			// After iterations are done, close the socket.
			clientSocket.close();
			
		}catch(Exception e){
			e.printStackTrace();
		}
		
		Date stop_time = new Date();
		double etime = (stop_time.getTime() - start_time.getTime())/1000.;
		System.out.println("\nElapsed Time = " + fixedWidthDoubletoString(etime,12,3) + " seconds\n");
	}
	
	// Used to print out the elapsed time whose unit is second.
	public static String fixedWidthDoubletoString (double x, int w, int d) {
		java.text.DecimalFormat fmt = new java.text.DecimalFormat();
		fmt.setMaximumFractionDigits(d);
		fmt.setMinimumFractionDigits(d);
		fmt.setGroupingUsed(false);
		String s = fmt.format(x);
		while (s.length() < w) {
			s = " " + s;
		}
		return s;
	}
	

}
