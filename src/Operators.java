import java.util.*;

import org.AMTL_Matrix.*;
import org.AMTL_Matrix.MatrixOps.*;
import org.AMTL_Matrix.Norms.Norms;


public class Operators {
	
	public double step_size;
	
	public Operators(double step_size){
		this.step_size = step_size;
	}
	
	/* *********************
	    Gradient Operators
	 * *********************/
	
	/* Square Loss 
	 * 1/2 ||A*w - b||^2, A is a nxd matrix, b is a nx1 vector and w is a dx1 vector
	   Returns: w - step_size* A' * (A*w - b) */
	
	public AMTL_Matrix SquareLoss_Forward(AMTL_Matrix A, AMTL_Matrix b, AMTL_Matrix w){
		
		AMTL_Matrix A_copy = new AMTL_Matrix(A);
		AMTL_Matrix b_copy = new AMTL_Matrix(b);
		
		AMTL_Matrix grad = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		AMTL_Matrix updated_point = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		AMTL_Matrix Aw = new AMTL_Matrix(b.NumRows, b.NumColumns, b.BlasID);
		AMTL_Matrix Aw_b = new AMTL_Matrix(b.NumRows, b.NumColumns, b.BlasID);
		
		MatrixOps.MULT(A_copy, w, Aw);
		MatrixOps.ReverseSign(b_copy);
		MatrixOps.ADD(Aw, b_copy, Aw_b);
		MatrixOps.Transpose(A_copy);
		MatrixOps.MULT(A_copy, Aw_b, grad);
		
		MatrixOps.Scale(grad, -step_size);
		MatrixOps.ADD(w, grad, updated_point);
		
		
		return updated_point;
	}
	
	/* Square Loss with l2 regularization
	 * 1/n ||A*w - b||^2 + (lambda/2)*||q||^2, A is a nxd matrix, b is a nx1 vector and w is a dx1 vector
	   Returns: q - step_size* ((2/n)* A' * (A*w - b) + lambda*q) */
	
	public AMTL_Matrix SquareLossReg_Forward(AMTL_Matrix X, AMTL_Matrix y, AMTL_Matrix p, AMTL_Matrix q, double reg){

		AMTL_Matrix X_copy = new AMTL_Matrix(X);
		AMTL_Matrix y_copy = new AMTL_Matrix(y);
		int num = y.NumRows; // number of data point
		AMTL_Matrix w = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // whole component
		MatrixOps.ADD(p, q, w); // w = p + q
		AMTL_Matrix Aw = new AMTL_Matrix(y.NumRows, y.NumColumns, y.BlasID);
		AMTL_Matrix Aw_b = new AMTL_Matrix(y.NumRows, y.NumColumns, y.BlasID);
		AMTL_Matrix tmp1 = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID); // store A' * (A*w - b) part
		AMTL_Matrix tmp2 = new AMTL_Matrix(q); // store q
		AMTL_Matrix q_new = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // store new q, q_{t}^{k}, all initially have the value of zero.

		MatrixOps.MULT(X_copy, w, Aw);
		MatrixOps.ReverseSign(y_copy);
		MatrixOps.ADD(Aw, y_copy, Aw_b);
		MatrixOps.Transpose(X_copy);
		MatrixOps.MULT(X_copy, Aw_b, tmp1);
		MatrixOps.Scale(tmp1, (2.0/num));
		MatrixOps.Scale(tmp2, reg);
		AMTL_Matrix sum = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		MatrixOps.ADD(tmp1, tmp2, sum);
		MatrixOps.Scale(sum, -step_size);
		MatrixOps.ADD(sum, q, q_new);		

		return q_new;
	}
	
	/* Compute output gradient
	   Returns: (2/n)* A' * (A*w - b) */
	
	public AMTL_Matrix SquareLossGrad_Forward(AMTL_Matrix X, AMTL_Matrix y, AMTL_Matrix p, AMTL_Matrix q){

		AMTL_Matrix X_copy = new AMTL_Matrix(X);
		AMTL_Matrix y_copy = new AMTL_Matrix(y);
		int num = y.NumRows; // number of data point
		AMTL_Matrix w = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // whole component
		MatrixOps.ADD(p, q, w); // w = p + q
		AMTL_Matrix Aw = new AMTL_Matrix(y.NumRows, y.NumColumns, y.BlasID);
		AMTL_Matrix Aw_b = new AMTL_Matrix(y.NumRows, y.NumColumns, y.BlasID);
		AMTL_Matrix q_new = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // store new q, q_{t}^{k}, all initially have the value of zero.

		MatrixOps.MULT(X_copy, w, Aw);
		MatrixOps.ReverseSign(y_copy);
		MatrixOps.ADD(Aw, y_copy, Aw_b);
		MatrixOps.Transpose(X_copy);
		MatrixOps.MULT(X_copy, Aw_b, q_new);
		MatrixOps.Scale(q_new, (2.0/num));

		return q_new;
	}
	
	/* Quadratic Loss
	 * 1/2 w' * Q * w + c' * w, Q is a dxd matrix, c is a dx1 vector and w is a dx1 vector
	   Returns: w - step_size* (Q * w + c) */
	
	public AMTL_Matrix Quadratic_Forward(AMTL_Matrix Q, AMTL_Matrix c, AMTL_Matrix w){
		
		AMTL_Matrix Q_copy = new AMTL_Matrix(Q);
		
		AMTL_Matrix grad = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		AMTL_Matrix Qw = new AMTL_Matrix(c.NumRows, c.NumColumns, c.BlasID);
		AMTL_Matrix updated_point = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		
		MatrixOps.MULT(Q_copy, w, Qw);
		MatrixOps.ADD(Qw, c, grad);
		
		MatrixOps.Scale(grad, -step_size);
		MatrixOps.ADD(w, grad, updated_point);
		
		return updated_point;
		
	}
	
	/* Log loss with l2 regularization (change)
	 * L(q) = \frac{1}{N} \sum_{i=1}^{N} \log (1+e^(-y_{i}*x_{i}^{T}*(p+q))) + \frac{\lambda}{2}\|q\|^{2}, x_i is a dx1 vector, y is a Nx1 vector and p and q are dx1 vector
	 * Returns: q - step_size * \frac{1}{N} \sum_{i=1}^{N}( (-y_{i}*x_{i}) e^(-y_{i}*x_{i}^{T}*(p+q))/(1+e^(-y_{i}*x_{i}^{T}*(p+q))) ) + \lambda * q
	 
	 * All the values in the following are using toy example 
	 */
	public AMTL_Matrix LogLossReg_Forward(AMTL_Matrix X, AMTL_Matrix y, AMTL_Matrix p, AMTL_Matrix q, double reg){
				
		AMTL_Matrix X_copy = new AMTL_Matrix(X); // X_copy is the same shape and value as X, but a different one!!!
		AMTL_Matrix y_copy = new AMTL_Matrix(y);
		AMTL_Matrix w = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // whole component, all zero when initialization 
		MatrixOps.ADD(p, q, w); // w = p + q

		double product = 0; // the value of obj_result
		double denominator; // the denominator part of gradient 
		int[] rows = new int[]{0}; // index of row, for data X
		int[] columns = new int[X_copy.NumColumns]; // index of columns, for extraction one row from data X, all zero here
		AMTL_Matrix q_new = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // store new q, q_{t}^{k}, all initially have the value of zero.
		AMTL_Matrix x = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // store one data point, one row of X
		AMTL_Matrix obj_result = new AMTL_Matrix(1,1,q.BlasID); // x^{T}(p+q)
		AMTL_Matrix sum = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // The running sum of loss functions

		MatrixOps.ReverseSign(y_copy); // y_{i} -> -y_{i}

		for(int i = 0; i<X_copy.NumColumns; i++){
			columns[i] = i;
		}

		for(int i = 0; i<X_copy.NumRows; i++){
						
			rows[0] = i;

			x = X_copy.getSubMatrix(rows, columns); // one data point, one row of X, but x now is a column vector because it is transposed in getSubMatrix

			MatrixOps.Transpose(x); // x_{i} -> x_{i}^{T}, x = (1,2)

			// Now w = (2,3)^{T}

			MatrixOps.MULT(x, w, obj_result); // obj_result = x_{i}^{T}*(p+q) = 8

			product = obj_result.getDouble(0, 0); // value of x_{i}^{T}*(p+q) = 8

			denominator = 1 / (1 + Math.exp(y_copy.getDouble(i, 0) * product)); // 1/( 1+e^(-y_{i}*x_{i}^{T}*(p+q)) )
			// Now denominator = 3.353501304664781E-4

			MatrixOps.Transpose(x); // x_{i}^{T} -> x_{i} x = (1,2)^{T}
			AMTL_Matrix vec = new AMTL_Matrix(x); // temporarily storage, vec = x_{i} = (1,2)^{T}
			MatrixOps.Scale(vec, y_copy.getDouble(i, 0)); // vec -> -y_{i}*x_{i}

			MatrixOps.Scale(vec, Math.exp(y_copy.getDouble(i, 0) * product)); // vec -> -y_{i}*x_{i}e^(-y_{i}*x_{i}^{T}*(p+q))
			// Now vec = (2.98E+03, 5.96E+03)^{T}
			MatrixOps.Scale(vec, denominator); // vec -> (-y_{i}*x_{i}) e^(-y_{i}*x_{i}^{T}*(p+q))/(1+e^(-y_{i}*x_{i}^{T}*(p+q)))
			MatrixOps.ADD(sum, vec, sum); // running sum

		}

		// Now sum = (2.99E+00, 3.99E+00)^{T}
		double R = X_copy.NumRows; // R = 3 
		MatrixOps.Scale(sum, (1.0 / R)); // average
		// Now sum = (9.98E-01, 1.33E+00)^{T}
		AMTL_Matrix q_tmp = new AMTL_Matrix(q); // temporarily store q because q will be scale in next step but need original value in last step
		MatrixOps.Scale(q, reg); // q -> \lambda * q, Now q = (2.00E+00, 4.00E+00)^{T}
		MatrixOps.ADD(q, sum, sum); // sum = (3.00E+00, 5.33E+00)^{T}
		MatrixOps.Scale(sum, -step_size); // sum = (-3.00E-02, -5.33E-02)^{T}
		MatrixOps.ADD(q_tmp, sum, q_new); // q_new = (9.70E-01, 1.95E+00)^{T}

		return q_new;
	}

	/* Compute output gradient
	 * Compare with LogLossReg_Forward, no regularization, no gradient descent step
	 * (change)
	 */
	public AMTL_Matrix LogLossGrad_Forward(AMTL_Matrix X, AMTL_Matrix y, AMTL_Matrix p, AMTL_Matrix q){
				
		AMTL_Matrix X_copy = new AMTL_Matrix(X); // X_copy is the same shape and value as X!!!
		AMTL_Matrix y_copy = new AMTL_Matrix(y);
		AMTL_Matrix w = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // whole component, all zero when initialization 
		MatrixOps.ADD(p, q, w); // w = p + q = (2,3)^{T}

		double product = 0; // the value of obj_result
		double denominator; // the denominator part of gradient 
		int[] rows = new int[]{0}; // index of row, for data X
		int[] columns = new int[X_copy.NumColumns]; // index of columns, for extraction one row from data X
		AMTL_Matrix grad = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // store new q, q_{t}^{k}, all initially have the value of zero.
		AMTL_Matrix x = new AMTL_Matrix(q.NumRows, q.NumColumns, q.BlasID); // store one data point, one row of X
		AMTL_Matrix obj_result = new AMTL_Matrix(1,1,q.BlasID); // x^{T}(p+q)

		MatrixOps.ReverseSign(y_copy); // y_{i} -> -y_{i} {1.0, -1.0, 1.0}

		for(int i = 0; i<X_copy.NumColumns; i++){
			columns[i] = i;
		}

		for(int i = 0; i<X_copy.NumRows; i++){
						
			rows[0] = i;

			x = X_copy.getSubMatrix(rows, columns); // one data point, one row of X
			MatrixOps.Transpose(x); // x_{i} -> x_{i}^{T}, x = (1,2)

			// Now w = (2,3)^{T}
			
			MatrixOps.MULT(x, w, obj_result); // obj_result = x_{i}^{T}*(p+q) = 8
			product = obj_result.getDouble(0, 0); // value of x_{i}^{T}*(p+q) = 8

			denominator = 1 / (1 + Math.exp(y_copy.getDouble(i, 0) * product)); // 1/( 1+e^(-y_{i}*x_{i}^{T}*(p+q)) )
			// Now denominator = 3.353501304664781E-4

			MatrixOps.Transpose(x); // x_{i}^{T} -> x_{i} x = (1,2)^{T}
			AMTL_Matrix vec = new AMTL_Matrix(x); // temporarily storage, vec = x_{i} = (1,2)^{T}
			MatrixOps.Scale(vec, y_copy.getDouble(i, 0)); // vec -> -y_{i}*x_{i}
			MatrixOps.Scale(vec, Math.exp(y_copy.getDouble(i, 0) * product)); // vec -> -y_{i}*x_{i}e^(-y_{i}*x_{i}^{T}*(p+q))
			// Now vec = (2.98E+03, 5.96E+03)^{T}
			MatrixOps.Scale(vec, denominator); // vec -> (-y_{i}*x_{i}) e^(-y_{i}*x_{i}^{T}*(p+q))/(1+e^(-y_{i}*x_{i}^{T}*(p+q)))
			MatrixOps.ADD(grad, vec, grad); // running sum of gradient
		}
		
		// Now grad = (2.99E+00, 3.99E+00)^{T}
		double R = X_copy.NumRows; // R = 3 
		MatrixOps.Scale(grad, (1.0 / R)); // average
		// Now grad = (9.98E-01, 1.33E+00)^{T}

		return grad;
	}
	
	
	/* Hinge loss
	 * \sum_{i}^{N} max(0,(1- b_i* a_i ' * w)), a_i is a dx1 vector, b is a Nx1 vector and w is a dx1 vector
	 */
	public AMTL_Matrix HingeLoss_Forward(AMTL_Matrix A, AMTL_Matrix b, AMTL_Matrix w){
		
		AMTL_Matrix A_copy = new AMTL_Matrix(A);
		AMTL_Matrix b_copy = new AMTL_Matrix(b);
		
		AMTL_Matrix updated_point = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		
		AMTL_Matrix vector = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
		AMTL_Matrix obj_result = new AMTL_Matrix(1,1,w.BlasID);
		double product = 0;
		
		MatrixOps.ReverseSign(b_copy);
		AMTL_Matrix sum = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);

		int[] rows = new int[]{0};
		int[] columns = new int[A_copy.NumColumns];
		for(int i = 0; i<A_copy.NumColumns; i++){
			columns[i] = i;
		}
		for(int i = 0; i<A_copy.NumRows; i++){
			rows[0] = i;
			vector = A_copy.getSubMatrix(rows, columns);
			MatrixOps.Transpose(vector);
			MatrixOps.MULT(vector, w, obj_result);
			product = b_copy.getDouble(i, 0) * obj_result.getDouble(0, 0);;
			
			if((1 + product) > 0){
				MatrixOps.Transpose(vector);
				MatrixOps.Scale(vector, b_copy.getDouble(i, 0));
				MatrixOps.ADD(sum, vector, sum);
			} else{
				AMTL_Matrix Zeros = new AMTL_Matrix(w.NumRows, w.NumColumns, w.BlasID);
				MatrixOps.ADD(sum, Zeros, sum);
			}
			
		}
		
		double R = A_copy.NumRows;
		
		MatrixOps.Scale(sum, (1 / R));
		MatrixOps.Scale(sum, -step_size);
		MatrixOps.ADD(w, sum, updated_point);
		
		return updated_point;
	}
	
	/* *********************
    Proximal Operators
 * *********************/
	
	/* Proximal operator of Trace norm ||W||_{*}
	 * Singular value thresholding, Input is a dxn model matrix where d is the dimension of the model
	 * and n is the number of tasks in multi-task learning setting.
	   Returns: U(S - (step_size * Lambda)*I)V' */
	
	public AMTL_Matrix  Prox_Trace(AMTL_Matrix Input, double mu){ // (change)
		
		//double threshold = step_size*Lambda;
		
		Norms.SingularValueThresholding(Input, mu);
		
		return Input;
	}
	
	/* Proximal operator of l1 norm ||w||_{1}
	 * Soft thresholding, Input is a dx1 model vector.
	   Returns: [prox(x)]_{i} = sign(x_i)max(|x_i| - (step_size * Lambda), 0) */
	
	public AMTL_Matrix Prox_l1(AMTL_Matrix Input, double Lambda){
		
		double val, x;
		for(int i = 0; i<Input.NumRows; i++){
			x = Input.getDouble(i, 0);
			val = Math.abs(x) - Lambda * step_size;
			if(val > 0){
				Input.setDouble(i, 0, ((x / Math.abs(x))*val));
			} else{
				Input.setDouble(i, 0, 0);
			}
		}
		
		return Input;
	}
	
	/* Proximal operator of l2 norm square 
	 * R(x) = Lambda/2 ||w||_{2}^{2}
	 * Returns: prox(x) = x / (1 + (step_size * Lambda) ) */
	
	public AMTL_Matrix Prox_l2_square(AMTL_Matrix Input, double Lambda){
		
		MatrixOps.Scale(Input, (1 / (1 + step_size * Lambda)));
		
		return Input;
	}
	
	/* Proximal operator of l2 norm  
	 * R(x) = Lambda ||w||_{2}, block soft thresholding
	 * Returns: prox(x) = w * (1 - (step_size * Lambda)/||w||_{2})_{+}*/
	
	public AMTL_Matrix Prox_l2(AMTL_Matrix Input, double Lambda){
		
		double val = Norms.L2_Norm(Input);
		double T = step_size*Lambda;
		
		if(val >= T){
			MatrixOps.Scale(Input, (1 - T/val));
		} else {
			MatrixOps.Scale(Input, 0);
		}
		
		return Input;
		
	}
	
	/* Proximal operator of elastic net 
	 * R(x) = Lambda1 ||w||_{1} + Lambda2/2 ||w||_{2}^{2}
	 * Returns: prox(x) = 1 / (1 + (step_size * Lambda2)) * prox_{l1,Lambda1} (w)*/
	
	public AMTL_Matrix Prox_ElasticNet(AMTL_Matrix Input, double Lambda1, double Lambda2){
		
		AMTL_Matrix prox = this.Prox_l1(Input, Lambda1);
		
		double val = 1 / (1 + (step_size * Lambda2));
		MatrixOps.Scale(prox, val);
		
		return prox;
	}
	

}
