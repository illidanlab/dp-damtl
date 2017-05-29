import java.io.*;
import java.net.*;
import java.util.*;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixIO;
import java.io.IOException;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.*;
import org.AMTL_Matrix.Norms.Norms;

public class Client_Log_LossTest {
	// Task node side, use logistic loss in classification problem, least square loss in regression problem
	/* Combined with Server_ProxTraceTest.java and ServerThread_TraceTest.java, these 3 .java file are test if asynchronous mechanism works in different computers scenario.
	 * In first test:
	 * in the only client, each element of the vector will plus one. 
	 * In the server, first each element in the column correspond to the client will plus one, then all the elements in the matrix will plus one
	 * In second test:
	 * in client 1, each element of the vector will plus one. in client 2, each element of the vector will plus 0.7. 
	 * In the server, all the elements in the matrix will plus one,
	 * Lab computer as central server and GPU as task node
	 * 
	 */
	private static int index = 1; // Which task node we are in
	private static int countITER = 0; // show the number of current iteration 
	private static double Lambda = Math.pow(10, -3); // regularization parameter
	private static double eps = 0.1; // privacy parameter
	private static double S1 = Math.pow(10, -4); // sensitivity on q_new, change according to alpha1
	private static double S2 = Math.pow(10, -2); // sensitivity on output gradient
	private static ArrayList<String> error = new ArrayList<String>(); // array of error rate in all iteration
	
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
		System.out.println("This is task node " + Integer.toString(index) + ", the error rate in iteration " + Integer.toString(countITER) + " is " + e);
		error.add(e); // ArrayList doesn't accept primitive numbers here
		
		DenseMatrix64F a = new DenseMatrix64F(1, error.size()); // in matrix form
		for(int i=0; i<error.size(); i++){
            a.set(0, i,  Double.parseDouble(error.get(i)));
		} 
		
		try{ // should write all error rate into a file whenever get a new error rate, it will overwrite the previous one
			MatrixIO.saveCSV(a,"errorRate"+Integer.toString(index)); // the file is stored under project/tm1
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
	}
	
	public static void measurer(AMTL_Matrix data, AMTL_Matrix label, AMTL_Matrix model){
		// Mean squared error of task model in each iteration, classification problem
		
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		//InputStream is;
		//InputStreamReader isr;
		//BufferedReader br;
		
		int ITER = 100; // Number of iterations (change), need proper number to converge
		double alpha1 = 0.01; // gradient descent (change), remember to change S1 when change alpha1, here we don't get ITER and alpha1 from user input
		
		
		// Blas is the ID that tells us which BLAS library we would like to use.
		/* BlasID 0: ejml
		 * BlasID 1: ujmp
		 * BlasID 2: jama
		 * BlasID 3: jblas
		*/
		int Blas = 0;
		
		// objective is logistic loss with l2 regularization. See operators. (change) from this line
		DenseMatrix64F X_load;
		
		try{ // data matrix X
			X_load = MatrixIO.loadCSV("data"+index); // "data"+index works
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		
		AMTL_Matrix X = new AMTL_Matrix(X_load,Blas); 
		System.out.println("Data set "+ Integer.toString(index) +" is loaded!");
		
		DenseMatrix64F y_load;
		
		try{ // label vector y
			y_load = MatrixIO.loadCSV("labelc"+index);
		}catch (IOException e2){
			throw new RuntimeException(e2);
		}
		
		AMTL_Matrix y = new AMTL_Matrix(y_load,Blas);
		System.out.println("Label set "+ Integer.toString(index) +" is loaded!");

		// Dimension of the feature vectors
		int dim = X.NumColumns; 
		dim = 4; // a test
		System.out.println("Dimension of the feature vector is " + Integer.toString(dim));
		/*
		// Read the initial q, use q_{t}^{0} in task t, do only one time for each task, for whole project 
		DenseMatrix64F q0;
		try{ // Works
			q0 = CommonOps.extract(MatrixIO.loadCSV("startQ"), index-1, index, 0, dim); // load corresponding row of task model matrix, note that when we use index data set, the task model should enter (index-1 to index), q0 is a row vector
		}catch (IOException e1){
			throw new RuntimeException(e1);
		}
		CommonOps.transpose(q0); // transform to column vector
		AMTL_Matrix q_ini = new AMTL_Matrix(q0,Blas); // row vector
		System.out.println("Task node "+ Integer.toString(index) +" initial q is loaded!");
				
		AMTL_Matrix p_new; // New p, received from central server 
		AMTL_Matrix q_old = new AMTL_Matrix(q0,Blas); ; // Old q, computed and left in this task node in last time, note the fact that it is equal to q0 now is just to initialize it to some value, it has nothing to do with q0, it will assign to other values later 
		AMTL_Matrix q_new; // New q 
		AMTL_Matrix grad; // Output gradient 
		*/
		// Creating an object of ClientMessage class. 
		ClientMessage clientMsg = new ClientMessage(dim,Blas);

		// This is a standard way to keep the time.
		Date start_time = null;
		
		try{
			// Set the socket
			//InetAddress serverHost = InetAddress.getByName("localhost"); // 127.0.0.1 in address.txt indicates the localhost (lab computer) itself
			InetAddress serverHost = InetAddress.getByName("35.9.42.254"); // task node need to know the ip address of central server
			
			// Server port number should be same as the number defined in Server_ProxTrace.java
			int serverPort = 3457;
			Socket clientSocket = null;
			
			// Start the timer
			start_time = new Date();			
			
			// Start work
			for(int j = 0; j < ITER; j++){
				System.out.println("ITER: " + j);
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
				oos.writeObject(clientMsg);
				oos.flush();

				// Get the message (new p) at the end of the operation at server's end.
				ois = new ObjectInputStream(clientSocket.getInputStream());
				clientMsg = (ClientMessage)ois.readObject( );

				if(clientMsg.getError() == 0){
					// Operation needs to be done at client end.	
					
					// test begin
					AMTL_Matrix test = new AMTL_Matrix(clientMsg.getVec());
					System.out.println("From central server we got: "); 
					System.out.println(test.M); 
					
					double[] v = {+0.7, +0.7, +0.7, +0.7}; // in the client, each element of the vector will plus 0.7.
					AMTL_Matrix a = new AMTL_Matrix(4, 1, Blas); 
					AMTL_Matrix b = new AMTL_Matrix(4, 1, Blas); 
					for (int i = 0; i < 4; i++) {
					    a.setDouble(i, 0, v[i]);
					}
					MatrixOps.ADD(a, test, b);
					
					System.out.println("Task node " + Integer.toString(index) + " send to central server: "); 
					System.out.println(b.M); 
					clientMsg.copyVec(b);
					// test end
					
					/*
					p_new = new AMTL_Matrix(clientMsg.getVec()); // New p, better to new a new AMTL_Matrix type
					System.out.println("From central server we got: "); // show what do we got from central server
					System.out.println(p_new.M); 
					
					// Generate two noise vectors
					/*
					DPNoise dpn1 = new DPNoise(dim, S1, eps);
					dpn1.compute();
					AMTL_Matrix Dpn1 = new AMTL_Matrix(dpn1.output(), 0); // Constructor: AMTL_Matrix(Object Input, int BlasID)
					DPNoise dpn2 = new DPNoise(dim, S2, eps);
					dpn2.compute();
					AMTL_Matrix Dpn2 = new AMTL_Matrix(dpn2.output(), 0);
					
					
					//Forward step 1 (Gradient Update on q)
					Operators forward1 = new Operators(alpha1);

					if(j == 0){
						q_new = forward1.LogLossReg_Forward(X, y, p_new, q_ini, Lambda); // First iteration q is initialized by the minimizer of L(q) with p=0
					}else{
						q_new = forward1.LogLossReg_Forward(X, y, p_new, q_old, Lambda); // Other iteration q is the previous one
					}
					
					// Add noise on q_new
					AMTL_Matrix q_new_n = new AMTL_Matrix(q_new.NumRows, q_new.NumColumns, q_new.BlasID);
					//MatrixOps.ADD(Dpn1, q_new, q_new_n);

					// task model performance 
					AMTL_Matrix w = new AMTL_Matrix(p_new.NumRows, p_new.NumColumns, p_new.BlasID);; // w = p_new + q_new_n, for both noisy and consistency consideration
					MatrixOps.ADD(p_new, q_old, w); // w = p + q, pay attention here p_new corresponding to q_old
					measurec(X, y, w);

					// Store for the use in next iteration, not q_new_n, see algorithm
					q_old = new AMTL_Matrix(q_new); // q_old has its own storage space
					
					// Forward step 2 (Compute output gradient)
					Operators forward2 = new Operators(alpha1); // StepSize alpha1 here is useless since we only use part of LogLossReg_Forward as LogLossGrad_Forward and there is no gradient update involved in it					
					grad = forward2.LogLossGrad_Forward(X, y, p_new, q_new_n); // p here should be p_new and q should be q_new_n

					// Add noise on grad
					AMTL_Matrix grad_n = new AMTL_Matrix(grad.NumRows, grad.NumColumns, grad.BlasID);
					//MatrixOps.ADD(Dpn2, grad, grad_n);
					
				    // Update client message with the updated vector.
					clientMsg.copyVec(grad_n); // Should send grad_n here
					*/
				
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
