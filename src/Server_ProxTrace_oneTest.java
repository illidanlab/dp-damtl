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
//import java.security.MessageDigest;
import org.ejml.ops.MatrixIO;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.MatrixOps;
import org.AMTL_Matrix.Norms.Norms;


import org.ejml.data.DenseMatrix64F;


public class Server_ProxTrace_oneTest {
	
	public static void main(String[] args){
		
		// index of the client port in the port list
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
			
			// Creating a socket by binding the port number. Server is ready to listen 
			// from this port.
			int serverPort = 3457; // (change)
			ServerSocket serverSocket = new ServerSocket(serverPort);
			
			System.out.println("****** Get Ready (Starts listening) ******");
			
			while(true){
				
				Socket clientSocket = serverSocket.accept();
				System.out.println("Starts communicating a client.");
				
				ServerThread_Trace_oneTest t = new ServerThread_Trace_oneTest(clientSocket, dim, W, StepSize, Lambda);
				
				FutureTask<AMTL_Matrix> ft = new FutureTask<AMTL_Matrix>(t);

				new Thread(ft).start();
				
				W = (AMTL_Matrix) ft.get();	
				System.out.println("The model matrix now is: "); 
				System.out.println(W.M);
			}
			
			
			
		} catch (Exception e){
			e.printStackTrace();
		}
	}
	
}