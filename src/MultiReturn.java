import org.AMTL_Matrix.AMTL_Matrix;


public class MultiReturn {
	
	AMTL_Matrix Q;
	AMTL_Matrix P; 
	AMTL_Matrix S;
	double v;
	
	public MultiReturn(AMTL_Matrix aa, AMTL_Matrix a, AMTL_Matrix b, double vv) {
	    this.Q = new AMTL_Matrix(aa); // it would be better if pass by value, not by reference  
	    this.P = new AMTL_Matrix(a);  
	    this.S = new AMTL_Matrix(b);
	    this.v = vv;
	}
	
	public static void main(String[] args){
		System.out.println("MultiReturn class!");
	}
}
