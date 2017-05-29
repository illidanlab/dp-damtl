import java.io.*;
import java.net.*;
import java.util.*;
import java.util.concurrent.FutureTask;
import java.io.IOException;
import java.io.PrintWriter;
import java.math.BigInteger;
//import java.security.MessageDigest;
import org.ejml.ops.MatrixIO;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.MatrixOps;
import org.AMTL_Matrix.Norms.Norms;


import org.ejml.data.DenseMatrix64F;


public class Server_ProxTraceTest {
	
	public static void main(String[] args){
				
		// Read the addresses of possible clients
		ReadAddress reader = new ReadAddress("/home/decs/Desktop/Javaws/DAMTLDP/address.txt");
		
		ArrayList<String> addressList = reader.readAddress();
		HashMap<String, BigInteger> addressSearch = reader.convertHash(addressList);
		
		// index of the client ip address in the address list
		int index = 0;
		
		/* BlasID 0: ejml
		 * BlasID 1: ujmp
		 * BlasID 2: jama
		 * BlasID 3: jblas
		*/
		
		
        int Blas = 0;
		
		//Initialization of the model matrix
        double[][] w = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}}; 
		AMTL_Matrix W = new AMTL_Matrix(w, Blas); 

		// Dimension of the model vector
		int dim = W.NumRows;
		System.out.println("Dimension of the feature vector is " + Integer.toString(dim));
		
		// Parsing command line arguments
		double StepSize = 0.5;
		double Lambda = 0.3;
		
		
		
		try {
			
			//Creating a socket by binding the port number. Server is ready to listen 
			// from this port.
			int serverPort = 3457;
			ServerSocket serverSocket = new ServerSocket(serverPort);
			
			System.out.println("****** Get Ready (Starts listening) ******");
			
			while(true){
				
				// accept(): A blocking method call. When I client contacts, the method is unblocked 
				// and returns a Socket object to the server to communicate with the client. 
				Socket clientSocket = serverSocket.accept();
				System.out.println("Starts communicating a client.");
				
				try{
					//
					InetAddress address = clientSocket.getInetAddress();
					System.out.println("Current client IP: " + address.getHostAddress());
					
					// This index will specify the column of the model matrix server needs to 
					// return.
					index = reader.searchIndex(address.getHostAddress(), addressSearch);
					
					// If there is a new Client, permission will be denied and it will be terminated.
					if (index == -1){
						System.out.println("New Client!");
					} else {
						System.out.println("Current index: " + index);
					}
				} catch(Exception ex){
					ex.printStackTrace();
				}
				
				ServerThread_TraceTest t = new ServerThread_TraceTest(clientSocket, dim, index, W, StepSize, Lambda);
				// FutureTask interface takes a callable object. Object ft is used to call the call() 
				// method overridden in ServerThread.
				FutureTask<AMTL_Matrix> ft = new FutureTask<AMTL_Matrix>(t);
				// This invokes the thread where call() method of ServerThread operates.
				new Thread(ft).start();
				// get() is a method of FutureTask and returns the result of 
				// the computations of call() method of ServerThread.
				W = (AMTL_Matrix) ft.get();	
				System.out.println("The model matrix now is: "); 
				System.out.println(W.M);	
			}
		} catch (Exception e){
			e.printStackTrace();
		}
	}
}