import org.AMTL_Matrix.AMTL_Matrix;

public class TwoReturn{
	AMTL_Matrix P; 
	AMTL_Matrix S;

	public TwoReturn(AMTL_Matrix a, AMTL_Matrix b) {
	    this.P = new AMTL_Matrix(a);  
	    this.S = new AMTL_Matrix(b);
	}
	
	public static void main(String[] args){
		System.out.println("TwoReturn class!");
	}
}


