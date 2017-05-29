import java.io.IOException;

public class Initialization {

	/**
	 * @param args
	 */

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
	    System.out.println("Call .py file from Java");
		String dir="/home/decs/Desktop/javaworkspace/project/test.py"; // Directory of Initialization.py
		String[] cmd = new String[2];
		cmd[0] = "python"; // check version of installed python: python -V
		cmd[1] = dir;
		try {
			Runtime rt = Runtime.getRuntime();
			Process pr = rt.exec(cmd);
			//Runtime.getRuntime().exec(cmd); // execute .py file
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		/*
		ProcessBuilder pb = new ProcessBuilder("python" + pyfile + ".py");
		pb.directory(new File(dir));
		pb.redirectError();
		Process p = pb.start();
		*/
		
	}
}
