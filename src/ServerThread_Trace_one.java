import java.io.*;
import java.net.*;
import java.util.concurrent.Callable;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.MatrixOps;
import org.AMTL_Matrix.Norms.Norms;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixIO;

public class ServerThread_Trace_one implements Callable<MultiReturn>{ // (change) the type in Callable<...> should be changed to TwoReturn
	
	Socket socket;
	
	AMTL_Matrix Q; // independent component of model matrix
	AMTL_Matrix P; // shared component of model matrix
	AMTL_Matrix S; // gradient matrix (w.r.t. p)
	AMTL_Matrix[] XX;
	AMTL_Matrix[] YY;
	int[] row_train;
	
	ClientMessage_one clientMsg;
	
	int index;
	int dim;
	double StepSize;
	double mu;
	double p_train=0.3;
	int Blas = 0;
	double value_all = 0.0; // overall objective function value
	
	
	public ServerThread_Trace_one(Socket clientSocket, AMTL_Matrix[] X, AMTL_Matrix[] Y, int[] r, int dim, AMTL_Matrix aa, AMTL_Matrix a, AMTL_Matrix b, double StepSize, double mu) {
		// TODO Auto-generated constructor stub
		this.socket = clientSocket;
		this.dim = dim;
		
		// Model matrix which was initialized by loading from a file at Server's end.
		Q = new AMTL_Matrix(aa);
		P = new AMTL_Matrix(a);
		S = new AMTL_Matrix(b);
		XX = X;
		YY = Y;
		row_train = r;
		//A_vec = new AMTL_Matrix(dim,1,a.BlasID);
		
		this.StepSize = StepSize;
		this.mu = mu; // here \mu is the \lambda in formula (2) in 2017KDD paper
		
	}

	public MultiReturn call() throws Exception { // (change) the type of return should be changed to TwoReturn
		
		// Get the message from the client
		ObjectInputStream ois = new ObjectInputStream(socket.getInputStream( ));
		// Client message was initialized in Client.java before. It has an initial vector which is 
		// the column of the model matrix that server needs to send back.
		clientMsg = (ClientMessage_one)ois.readObject( );
		this.index = clientMsg.getId();
		
		if(index == -1){
			clientMsg.setError(1);
		} else if(clientMsg.getVec().NumRows == dim){
			// New q vector carried by clientMsg.
			AMTL_Matrix nq = new AMTL_Matrix(clientMsg.getQnew()); 
						
			for(int i = 0; i<dim; i++){
				Q.setDouble(i, index, nq.getDouble(i,0)); 
			}
						
			// Initial vector carried by clientMsg.
			AMTL_Matrix grad = new AMTL_Matrix(clientMsg.getVec()); // gradient vector
						
			// Change the corresponding column of the gradient matrix S with the new gradient vector sent from task node, make sure use index 
			for(int i = 0; i<dim; i++){
				S.setDouble(i, index, grad.getDouble(i,0)); // update S 
			}
			
			// Operations need to be done by server.
			AMTL_Matrix T = new AMTL_Matrix(S); // store \P^{(k-1)}-\alpha_2 \S^{(k)}
			MatrixOps.Scale(T, -StepSize);
			MatrixOps.ADD(P, T, T);
			Operators backward = new Operators(StepSize);
			System.out.println("Rank of T is: " + MatrixOps.getRank(T));
			AMTL_Matrix P_new = backward.Prox_Trace(T, mu);
			System.out.println("Rank of P_new is: " + MatrixOps.getRank(P_new));
			
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
			
			/*
			// test begin, regression 
			// compute overall objective function value and store it, least square loss
			AMTL_Matrix W = new AMTL_Matrix(P.NumRows, P.NumColumns, P.BlasID); // newest model matrix
			MatrixOps.ADD(Q, P, W); // W = P + Q
			for(int i = 0; i<W.NumColumns; i++){ // for each TN
				// objective is least square loss with l2 regularization here
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
					value_each = value_each + Math.pow(product-YY[i].getDouble(j, 0),2); 				
				}
				
				value_each = value_each/row_train[i];
				value_all = value_all + value_each;
			}
			value_all = value_all + mu*Norms.Trace_Norm(W); // add unclear norm at the end
			// test end, regression
			*/	
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
		
		MultiReturn next = new MultiReturn(Q, P, S, value_all);
		return next; // P and S must be return in order to be used in next iteration
	}

}
