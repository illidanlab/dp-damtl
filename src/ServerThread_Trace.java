import java.io.*;
import java.net.*;
import java.util.concurrent.Callable;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.MatrixOps;

public class ServerThread_Trace implements Callable<TwoReturn>{ // (change) the type in Callable<...> should be changed to TwoReturn
	
	Socket socket;
	
	AMTL_Matrix P; // shared component of model matrix
	AMTL_Matrix S; // gradient matrix (w.r.t. p)
	
	ClientMessage clientMsg;
	
	int index;
	int dim;
	double StepSize;
	double mu;
	
	
	public ServerThread_Trace(Socket clientSocket, int dim, int index, AMTL_Matrix a, AMTL_Matrix b, double StepSize, double mu) {
		// TODO Auto-generated constructor stub
		this.socket = clientSocket;
		this.dim = dim;
		this.index = index; 
		
		// Model matrix which was initialized by loading from a file at Server's end.
		P = new AMTL_Matrix(a);
		S = new AMTL_Matrix(b);
		//A_vec = new AMTL_Matrix(dim,1,a.BlasID);
		
		this.StepSize = StepSize;
		this.mu = mu;
		
	}

	public TwoReturn call() throws Exception { // (change) the type of return should be changed to TwoReturn
		
		// Get the message from the client
		ObjectInputStream ois = new ObjectInputStream(socket.getInputStream( ));
		// Client message was initialized in Client.java before. It has an initial vector which is 
		// the column of the model matrix that server needs to send back.
		clientMsg = (ClientMessage)ois.readObject( );

		
		if(index == -1){
			clientMsg.setError(1);
		} else if(clientMsg.getVec().NumRows == dim){
			// Initial vector carried by clientMsg.
			AMTL_Matrix grad = new AMTL_Matrix(clientMsg.getVec()); // gradient vector
			
			// Change the corresponding column of the gradient matrix S with the new gradient vector sent from task node, index 
			for(int i = 0; i<dim; i++){
				S.setDouble(i, index, grad.getDouble(i,0)); // update S 
			}
			
			// Operations need to be done by server.
			AMTL_Matrix T = new AMTL_Matrix(S); // store \P^{(k-1)}-\alpha_2 \S^{(k)}
			MatrixOps.Scale(T, -StepSize);
			MatrixOps.ADD(P, T, T);
			Operators backward = new Operators(StepSize);
			AMTL_Matrix P_new = backward.Prox_Trace(T, mu);
			
			for(int i = 0; i<P_new.NumRows; i++){
				for(int j = 0; j<P_new.NumColumns; j++){
					P.setDouble(i, j, P_new.getDouble(i,j)); // update P 
				}
			}
			
			for(int i = 0; i<dim; i++){
				grad.setDouble(i,0, P.getDouble(i,index)); // extract that column and send back to corresponding task
			}
			
			// Updated vector is copied back to clientMsg.
			clientMsg.copyVec(grad);
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
		
		TwoReturn next = new TwoReturn(P, S);
		return next; // P and S must be return in order to be used in next iteration
	}

}


