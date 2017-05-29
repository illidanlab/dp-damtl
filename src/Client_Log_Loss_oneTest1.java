import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.TimeUnit;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixIO;
import java.io.IOException;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.*;
import org.AMTL_Matrix.Norms.Norms;

public class Client_Log_Loss_oneTest1 {
	/* Combined with Server_ProxTrace_oneTest.java and ServerThread_Trace_oneTest.java, these 3 .java file are test if asynchronous mechanism works in one computers scenario.
	 * different clients plus different value and change "System.out.println("Task node 1, ITER: " + j);"
	 * In the server, all the elements in the matrix will plus one,
	 * 
	 * Can also collaborate with Server_ProxTrace_one_sync_test.java
	 * 
	 */
	// Task node side, use logistic loss in classification problem, least square loss in regression problem
	private static int serverPort = 3457;
	private static int index; // Which task node we are in
	private static int latency; // wait as time delay in network, in MILLISECONDS, here use int
	private static int ITER = 20; // Number of iterations (change)
	private static int Blas = 0;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	
		// The index, latency parameters are read from outside input
		index = Integer.parseInt(args[0]);
		latency = Integer.parseInt(args[1]);
		
		System.out.println("Task node index is: " + index);
		System.out.println("Task node latency is: " + latency);
		
		int dim = 4; // a test
		System.out.println("Dimension of the feature vector is " + Integer.toString(dim));
	
		// Creating an object of ClientMessage_one class. 
		ClientMessage_one clientMsg = new ClientMessage_one(dim,Blas);

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
				System.out.println("Task node 1, ITER: " + j);
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
					clientMsg.copyId(index-1);
					// test end
					
				
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
