import java.io.IOException;
import java.util.*;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.ops.MatrixIO;
import org.ejml.ops.NormOps;
import org.apache.commons.math3.distribution.*; // in commons-math3-3.6.1 lib


public class DPNoise {

	/**
	 * @param args
	 */
	
	private int d; // data dimension
	private double S; // sensitivity
	private double eps; // privacy parameter
	private DenseMatrix64F noisev; // noise vector
	
	DPNoise(int d, double s, double eps){
		// Constructor, d for dimension of data, s for sensitivity, eps for privacy parameter
		this.d = d;
		this.S = s;
		this.eps = eps;
	}
	
	public void compute(){
		// compute noise
		
		double[] mean = new double[d]; // mean vector
		Arrays.fill(mean, 0);
	    double[][] matrix = new double[d][d]; // covariance matrix, an identity matrix
	    for(int i=0; i<d; i++){ 
			for(int j=0; j<d; j++){
				if(i == j){
					matrix[i][j] = 1;
				}else{
					matrix[i][j] = 0;
				}
			}
		}
	    
	    MultivariateNormalDistribution mnd = new MultivariateNormalDistribution(mean, matrix);
	    double[][] n = new double[d][1]; // store noise vector, in d*1 matrix form
	    double[] m = mnd.sample(); // sample a random vector
	    for(int j=0; j<d; j++){
	    	n[j][0] = m[j];
		} 
	    
		DenseMatrix64F noise = new DenseMatrix64F(n); // noise vector
		CommonOps.divide(NormOps.normF(noise), noise); // normalization 
		GammaDistribution gd = new GammaDistribution((double)d, S/eps); // beta = eps/S, scale = 1/beta = S/eps
		double norm = gd.sample(); // sample norm of noise vector
		CommonOps.scale(norm, noise); // output noise vector = norm * noise vector
		noisev = noise;
	}
	
	public DenseMatrix64F output(){
		//Output noise vector
		return noisev;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/*
		DPNoise dpn = new DPNoise(28, 0.0001, 0.1);
		dpn.compute();
		System.out.println(dpn.output());
		*/
	}

}