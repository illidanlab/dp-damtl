import java.io.*;
import java.net.*;
import java.util.concurrent.Callable;

import org.AMTL_Matrix.*;

public class ServerThread_TraceTest implements Callable<AMTL_Matrix>{
	
Socket socket;
	
	AMTL_Matrix W;
	
	ClientMessage clientMsg;
	
	int index;
	int dim;
	double StepSize;
	double Lambda;
	
	
	public ServerThread_TraceTest(Socket clientSocket, int dim, int index, AMTL_Matrix a, double StepSize, double Lambda) {
		// TODO Auto-generated constructor stub
		this.socket = clientSocket;
		this.dim = dim;
		this.index = index;
		
		// Model matrix which was initialized by loading from a file at Server's end.
		W = new AMTL_Matrix(a);
		//A_vec = new AMTL_Matrix(dim,1,a.BlasID);
		
		this.StepSize = StepSize;
		this.Lambda = Lambda;
		
	}

	public AMTL_Matrix call() throws Exception {
		
		// Get the message from the client
		ObjectInputStream ois = new ObjectInputStream(socket.getInputStream( ));
		// Client message was initialized in Client.java before. It has an initial vector which is 
		// the column of the model matrix that server needs to send back.
		clientMsg = (ClientMessage)ois.readObject( );

		
		if(index == -1){
			clientMsg.setError(1);
		} else if(clientMsg.getVec().NumRows == dim){
			// Initial vector carried by clientMsg.
			AMTL_Matrix A_vec = new AMTL_Matrix(clientMsg.getVec());
			System.out.println("From task node " + Integer.toString(index) + " we got: "); 
			System.out.println(A_vec.M); 
			
			// Change the corresponding column of the model matrix with the vector 
			// in the client message.
			
			for(int i = 0; i<dim; i++){
				W.setDouble(i, index, A_vec.getDouble(i,0)); // copy that column
			}
			
			
			for(int i = 0; i<W.NumRows; i++){
				for(int j = 0; j<W.NumColumns; j++){
					W.setDouble(i, j, W.getDouble(i,j)+1.0); // all the elements in the matrix will plus one
				}
			}
			
			
			for(int i = 0; i<dim; i++){
				A_vec.setDouble(i,0, W.getDouble(i,index));
			}
			
			System.out.println("Central server send to task node " + Integer.toString(index)); 
			System.out.println(A_vec.M);
			// Updated vector is copied back to clientMsg.
			clientMsg.copyVec(A_vec);
		} else{
			System.out.println("The vector of client" + index + "does not match the row number of the matrix!\n Permission Denied\n");
			clientMsg.setError(3);
		}
		
		// Serialize the clientMsg
		ObjectOutputStream oos = new ObjectOutputStream(socket.getOutputStream());
		oos.writeObject(clientMsg);
		oos.flush();
		
		// Close the socket one iteration is done.
		socket.close();
	
		return W;
	}

}