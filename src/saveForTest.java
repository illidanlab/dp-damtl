import org.AMTL_Matrix.AMTL_Matrix;
import org.AMTL_Matrix.MatrixOps.MatrixOps;
import org.AMTL_Matrix.Norms.Norms;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;


public class saveForTest {
	//This file save code for test use


	public static void test(){
		// test reference assign in Java, use AMTL_Matrix
		double[][] a = {{1,2},{3,4}};
		double[][] b = {{1,2,6},{3,4,8}};
		AMTL_Matrix A = new AMTL_Matrix(a,0); 
		AMTL_Matrix B = new AMTL_Matrix(b,0); 
		System.out.println(A.M);
		System.out.println(B.M);
		B = A;
		System.out.println(A.M);
		System.out.println(B.M);
	}
	
	//System.out.println(Arrays.toString(low1[1])); // print out one line of 2d array
	//System.out.println(Arrays.deepToString(low1)); // print out whole 2d array
	//System.out.println(res); // print out DenseMatrix64F type, ArrayList
	//System.out.println(X.M); // print out the actual content of a AMTL_Matrix type
	
	/*
		test CommonOps.insert
		double[][] x = {{1, 2, 2}, {2, 2, 1}, {2, 1, 2}};
		double[][] y = {{-1.0}, {+1.0}, {-1.0}};
		DenseMatrix64F X = new DenseMatrix64F(x);
		DenseMatrix64F Y = new DenseMatrix64F(y);
		CommonOps.insert(Y, X, 0, 2);
		
	*/
	/* measurer method
	 	double[][] p = {{1}, {1}, {1}};
		double[][] x1 = {{1, 2, 3}, {5, 3, 1}, {2, 1, 4}};
		double[][] y1 = {{-1.0}, {+1.0}, {-1.0}};
		AMTL_Matrix P = new AMTL_Matrix(p,0);
		AMTL_Matrix X = new AMTL_Matrix(x1,0);
		AMTL_Matrix Y = new AMTL_Matrix(y1,0);
		measurer(X, Y, P);
	 */
	
	/* test zscore method in Client_Square_Loss_one.java
		double[][] y = {{-1.0}, {+2.0}, {-1.0}, {+2.0}, {+2.0}, {+9.0}};
		DenseMatrix64F y_load = new DenseMatrix64F(y); 
		System.out.println(zscore(y_load));
	 */
	
	/* test compute overall objective function value in ServerThread_Trace_one.java code between "// test begin, regression" and "// test end, regression"
		double[][] p = {{1, 2}, {2, 3}, {3, 4}};
		double[][] q = {{2, 2}, {3, 3}, {4, 4}};
		double[][] x1 = {{1, 2, 3}, {2, 3, 1}, {2, 1, 4}};
		double[][] x2 = {{1, 4, 4}, {4, 4, 2}, {1, 3, 4}};
		double[][] y1 = {{-1.0}, {+1.0}, {-1.0}};
		double[][] y2 = {{+1.0}, {-1.0}, {+1.0}};
		AMTL_Matrix[] X = new AMTL_Matrix[2];
		AMTL_Matrix[] Y = new AMTL_Matrix[2];
		X[0] = new AMTL_Matrix(x1, 0);
		X[1] = new AMTL_Matrix(x2, 0);
		Y[0] = new AMTL_Matrix(y1, 0);
		Y[1] = new AMTL_Matrix(y2, 0);
		AMTL_Matrix P = new AMTL_Matrix(p,0);
		AMTL_Matrix Q = new AMTL_Matrix(q,0);
		int dim = 3;
		int[] row_train = {3,3};
		double value_all = 0.0; // overall objective function value

	*/
	
	/* test SquareLossReg_Forward and SquareLossGrad_Forward
		double[][] InputX = {{1, 2, 3}, {1, 3, 4}, {1, 4, 5}};
		AMTL_Matrix X_train = new AMTL_Matrix(InputX, 0);
		double[] Inputy = {-1.0, +1.0, -1.0}; // type: n_{t} * 1, one label in one row
		AMTL_Matrix y_train = new AMTL_Matrix(3, 1, 0);
		for (int i = 0; i < 3; i++) {
			y_train.setDouble(i, 0, Inputy[i]);
		}
		double[] Inputp = {1.0, 1.0, 2.0};
		double[] Inputq = {1.0, 2.0, 1.0};
		AMTL_Matrix p = new AMTL_Matrix(3, 1, 0);
		AMTL_Matrix q = new AMTL_Matrix(3, 1, 0);
		for (int i = 0; i < 3; i++) {
		    p.setDouble(i, 0, Inputp[i]);
		    q.setDouble(i, 0, Inputq[i]);
		}
		AMTL_Matrix q_new;
		
		Operators forward1 = new Operators(alpha1);
		q_new = forward1.SquareLossReg_Forward(X_train, y_train, p, q, 1.0); 
	 
	*/
	
	/* test compute overall objective function value in ServerThread_Trace_one.java code between "// test begin, classification" and "// test end, classification"
    double value_all = 0.0; 
	double mu = 0.0015;
	double[][] p = {{1, 2}, {2, 3}, {3, 4}};
	double[][] q = {{2, 2}, {3, 3}, {4, 4}};
	double[][] x1 = {{1, 2, 2}, {2, 2, 1}, {2, 1, 2}};
	double[][] x2 = {{3, 4, 4}, {4, 4, 3}, {4, 3, 4}};
	double[][] y1 = {{-1.0}, {+1.0}, {-1.0}};
	double[][] y2 = {{+1.0}, {-1.0}, {+1.0}};
	AMTL_Matrix P = new AMTL_Matrix(p,0);
	AMTL_Matrix Q = new AMTL_Matrix(q,0);
	AMTL_Matrix W = new AMTL_Matrix(P.NumRows, P.NumColumns, P.BlasID); // newest model matrix
	MatrixOps.ADD(Q, P, W); // W = P + Q

	for(int i = 0; i<W.NumColumns; i++){ // for each task
		double value_each = 0.0; // objective function value of each task
		DenseMatrix64F X_load;
		if (i == 0){
			X_load = new DenseMatrix64F(x1);
		}else{
			X_load = new DenseMatrix64F(x2);
		}
		int row_train = X_load.numRows;
		int dim = X_load.numCols;
		AMTL_Matrix X_train = new AMTL_Matrix(X_load,0);	

		DenseMatrix64F y_load;
		if (i == 0){
			y_load = new DenseMatrix64F(y1);
		}else{
			y_load = new DenseMatrix64F(y2);
		}
		AMTL_Matrix y_train = new AMTL_Matrix(y_load,0);

		int[] rows = new int[]{0}; 
		int[] columns = new int[dim];
		for(int k = 0; k<dim; k++){
			columns[k] = k;
		}

		for(int j = 0; j<row_train; j++){
			rows[0] = j;
			AMTL_Matrix obj_result = new AMTL_Matrix(1,1,0); // x_{i}^{T}*w
			double product = 0; // the value of obj_result
			AMTL_Matrix w = new AMTL_Matrix(CommonOps.extract((DenseMatrix64F) W.M, 0, row_train, i, i+1), 0);
			AMTL_Matrix x = new AMTL_Matrix(X_train.getSubMatrix(rows, columns)); // copy a new one to Transpose
			MatrixOps.Transpose(x);
			MatrixOps.MULT(x, w, obj_result);
			product = obj_result.getDouble(0, 0); // value of x_{i}^{T}*w
			value_each = value_each + Math.log( 1 + Math.exp(-1 * y_train.getDouble(j, 0) * product) ); // log ( 1+e^(-y_{i}*x_{i}^{T}*w) ), log is e based					
		}
		value_each = value_each/row_train;
		value_all = value_all + value_each;
	}
	value_all = value_all + mu*Norms.Trace_Norm(W); // add unclear norm at the end
	  
	*/
	
	/* A test input and output for LogLossReg_Forward, the test is recommended to be done line by line 
	 * Same for testing measurec function in Client_Log_Loss.java
	 * First time check logic 
	 * Second time check value
	 * // Toy example
		double reg = 2.0;
		double StepSize = 0.01; // gradient descent (change)
		double[][] p = {{1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}};
		double[][] q = {{2, 2, 2, 2}, {3, 3, 3, 3}, {4, 4, 4, 4}};
		double[][] p = {{1, 2}, {2, 3}, {3, 4}};
		double[][] q = {{2, 2}, {3, 3}, {4, 4}};
		double[][] x1 = {{1, 2, 2}, {2, 2, 1}, {2, 1, 2}};
		double[][] x2 = {{3, 4, 4}, {4, 4, 3}, {4, 3, 4}};
		double[][] y1 = {{-1.0}, {+1.0}, {-1.0}};
		double[][] y2 = {{+1.0}, {-1.0}, {+1.0}};
		double[][] InputX = {{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
		double[][] InputX = {{1, 2}, {1, 1}, {2, 2}}; // type: n_{t} * d matrix, one data in one row
		double[] Inputy = {-1.0, +1.0, -1.0}; // type: n_{t} * 1, one label in one row
		double[] Inputp = {1.0, 1.0, 2.0};
		double[] Inputp = {1.0, 1.0};
		double[] Inputq = {1.0, 2.0, 1.0};
		double[] Inputq = {1.0, 2.0};
		AMTL_Matrix X = new AMTL_Matrix(InputX, 0);
		AMTL_Matrix y = new AMTL_Matrix(3, 1, 0);
		AMTL_Matrix p = new AMTL_Matrix(2, 1, 0);
		AMTL_Matrix q = new AMTL_Matrix(2, 1, 0);

		for (int i = 0; i < 3; i++) {
		    y.setDouble(i, 0, Inputy[i]);
		}
		for (int i = 0; i < 2; i++) {
		    p.setDouble(i, 0, Inputp[i]);
		    q.setDouble(i, 0, Inputq[i]);
		}
				
		Operators forward = new Operators(StepSize);
		AMTL_Matrix q_new = forward.LogLossReg_Forward(X, y, p, q, reg);
	 */
	
	/* test GammaDistribution
	GammaDistribution gd = new GammaDistribution(100, 0.001);
	DenseMatrix64F test = new DenseMatrix64F(10000,1);
	for(int j=0; j<10000; j++){
		test.set(j, 0, gd.sample()); // sample norm of noise vector
	}
	try{
		MatrixIO.saveCSV(test, "gamma"); 
	}catch (IOException e1){
		throw new RuntimeException(e1);
	}
	*/
}
