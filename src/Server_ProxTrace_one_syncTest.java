/* Files end with "_one" can Run on one computer using shell script
 * The port, index, latency parameters are read from outside input
 * 
 * This file is to test synchronous mechanism, together with Client_Log_Loss_oneTesti
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


public class Server_ProxTrace_one_syncTest {
	
	private static AMTL_Matrix W;
	
	public static void main(String[] args){
		
		/* BlasID 0: ejml
		 * BlasID 1: ujmp
		 * BlasID 2: jama
		 * BlasID 3: jblas
		*/
		int T = 3; 
		int dim = 4; 
		int Blas = 0;
		int index; // index we received from specific task node
		ArrayList<Integer> tnlist = new ArrayList<Integer>(); // Remember the index of the task node that haven't send gradient to central server 
		for(int i = 0; i<T; i++){ 
			tnlist.add(i);
		}
		ArrayList<ClientMessage_one> clientMsglist = new ArrayList<ClientMessage_one>(); // for each task node, we have one message for it
		ArrayList<Socket> socketlist = new ArrayList<Socket>(); // one socket for each task 

		W = new AMTL_Matrix(dim, T, Blas); // initialize W, all zero
		
		try {
			
			// Creating a socket by binding the port number. Server is ready to listen 
			// from this port.
			int serverPort = 3457; // (change)
			ServerSocket serverSocket = new ServerSocket(serverPort);
			
			System.out.println("****** Get Ready (Starts listening) ******");
			
			
			while(true){ 
				Socket clientSocket = serverSocket.accept(); 
				socketlist.add(clientSocket); // a new socket (task)
				ObjectInputStream ois = new ObjectInputStream(clientSocket.getInputStream( ));
				clientMsglist.add((ClientMessage_one)ois.readObject( )); // ClientMessage_one contains the info of where is this message from, need corresponding socket to 
				index = (clientMsglist.get(clientMsglist.size()-1)).getId(); // get the last message's index, in each loop, one and only one grad is received
				tnlist.remove(new Integer(index)); // remove by value, not index
				AMTL_Matrix grad = new AMTL_Matrix((clientMsglist.get(clientMsglist.size()-1)).getVec()); // gradient vector
				System.out.println("From task node " + Integer.toString(index) + " we got: "); 
				System.out.println(grad.M); 
				System.out.println("Now task nodes " + tnlist + " still not send their gradient yet.");
				for(int i = 0; i<dim; i++){ 
					W.setDouble(i, index, grad.getDouble(i,0)); // copy that column
				}
				System.out.println("Now W at central server side is:"); 
				System.out.println(W.M); 
				
				if (tnlist.isEmpty()){ // grads from all tasks are received, process on them is then performed, 
					// the central server is now blocked from outside world
					for(int i = 0; i<W.NumRows; i++){
						for(int j = 0; j<W.NumColumns; j++){
							W.setDouble(i, j, W.getDouble(i,j)+1.0); // all the elements in the matrix will plus one
						}
					}
					
					for(int i = 0; i<T; i++){
						for(int j = 0; j<dim; j++){
							grad.setDouble(j,0, W.getDouble(j,i));
						}
						ObjectOutputStream oos = new ObjectOutputStream(socketlist.get(i).getOutputStream());
						clientMsglist.get(i).copyVec(grad);
						System.out.println("Now we send task node " + Integer.toString(i)); 
						System.out.println(clientMsglist.get(i).getVec().M); 
						oos.writeObject(clientMsglist.get(i));
						oos.flush();
						socketlist.get(i).close();
					}
					
					for(int i = 0; i<T; i++){ // renew 
						tnlist.add(i);
					}
					clientMsglist.clear(); // remove all elements
					socketlist.clear();
				}
			}
			
		} catch (Exception e){
			e.printStackTrace();
		}
	}
}
