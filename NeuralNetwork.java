import java.io.Serializable;
import java.io.*;
import MatrixMath.*;
public class NeuralNetwork implements Serializable { 
    /**
     *
     */
    private static final long serialVersionUID = 1L;
  Matrix I;
  Matrix IH;
  Matrix[] HH;
  Matrix[] H;
  Matrix HO;
  Matrix O;
  int[] arc = {7, 4, 3, 2};
  int fitness=0;
  boolean[] activation = new boolean[]{false,false,false,false,false,false};//{No activation, tanh, sin, cos, ln,sigmoid}
  int activationindex = 0;
  float learningrate = 0.1f;
  public NeuralNetwork(int[] arc, int activation) {
    this.activation[activation] = true;
    activationindex=activation;
    this.arc = arc;
    I = new Matrix(arc[0], 1);
    IH = new Matrix(arc[1], arc[0]);
    IH.randomize();
    IH= Matrix.rAdd(IH);
    HH = new Matrix[arc.length-2];
    H = new Matrix[arc.length-2];
    for (int i = 0; i < arc.length-2; i++) {
      H[i] = new Matrix(arc[i+1], 1);
      HH[i] = new Matrix(arc[i+2], arc[i+1]);
      HH[i].randomize();
      HH[i] = Matrix.rAdd(HH[i]);
    }
    HO = new Matrix(arc[arc.length-1], arc[arc.length-2]);
    HO.randomize();
    HO = Matrix.rAdd(HO);
    O = new Matrix(HO.m, 1);
  }
  static void serialize(NeuralNetwork NeuralNetwork) {
    if (NeuralNetwork.fitness > deserialize().fitness) {
      try {
        
        FileOutputStream fileOut =
        new FileOutputStream("bestNeuralNetwork.NeuralNetwork");
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(NeuralNetwork);
        out.close();
        fileOut.close();
        System.out.printf("Serialized data is saved in bestNeuralNetwork.NeuralNetwork");
      } 
      catch (IOException i) {
        i.printStackTrace();
        return;
      }
      catch(NullPointerException n) {
        return;
      }
    }
  }
  static NeuralNetwork deserialize() {
    int[] arc = {7, 4, 3, 2};
    NeuralNetwork NeuralNetwork=null;
    try {
      FileInputStream fileIn = new FileInputStream("bestNeuralNetwork.NeuralNetwork");
      ObjectInputStream in = new ObjectInputStream(fileIn);
      NeuralNetwork = (NeuralNetwork) in.readObject();
      in.close();
      fileIn.close();
    } 
    catch (IOException i) {
      //i.printStackTrace();
      return new NeuralNetwork(arc, 1);
    } 
    catch (ClassNotFoundException c) {
      //c.printStackTrace();
      return new NeuralNetwork(arc, 1);
    }
    //System.out.println("NeuralNetwork loaded"+" It's fitness is "+NeuralNetwork.fitness);
    return NeuralNetwork;
  }
  void mutate() {
    IH=Matrix.rAdd(IH); 
    for (int i = 0; i < HH.length; i++) {
      HH[i]=Matrix.rAdd(HH[i]);
    }
    HO=Matrix.rAdd(HO);
  }
  float[] ff(float[] inp) {
      int index = 0;
      for(int i = 0; i < activation.length; i++)
      {
          if(activation[i]){
              index = i;
              break;
          }
      }
    for (int i = 0; i < inp.length; i++) {
      I.table[i][0] = inp[i];
    }
    H[0] = Matrix.mMult(IH, I,index);
    for (int i = 1; i < HH.length; i++) {
      H[i] = Matrix.tanh(H[i]);
      H[i] = Matrix.mMult(HH[i-1], H[i-1],index);
    }
    O = Matrix.mMult(HO, H[H.length-1],index);
    O = Matrix.tanh(O);
    float[] out = new float[O.m];
    for (int i = 0; i < O.m; i++) {
      out[i]= O.table[i][0];
    }
    return out;
  }

  public static NeuralNetwork backPropogate(NeuralNetwork nn,float[] inp,float[] correct){
    Matrix corr = new Matrix(correct.length,1);
    Matrix[] errors  = new Matrix[nn.H.length+2];
    NeuralNetwork temp = new NeuralNetwork(nn.arc, nn.activationindex);
    float[] ans = nn.ff(inp);
    for(int i = 0; i < correct.length; i++){
        corr.table[i][0] = correct[i];
    }
    Matrix errorsO = Matrix.mSub(corr, nn.O);
	Matrix transposedHO = Matrix.transpose(nn.HO);
    Matrix errorsHLast = Matrix.mMult(transposedHO, errorsO,0);
    //Hidden to hidden
    Matrix[] errorsH = new Matrix[nn.H.length];
    errorsH[nn.H.length - 1] = errorsHLast;
    for (int n = nn.H.length - 2; n >= 0; n--) {
        Matrix transposedHH = Matrix.transpose(nn.HH[n]);
        errorsH[n] = Matrix.mMult(transposedHH, errorsH[n + 1],0);
    }
    // Adjust weights
	// Output to hidden
    Matrix gradient = new Matrix(nn.O.m, nn.O.n);
    gradient = Matrix.derive(gradient, nn.activationindex);
    gradient = Matrix.mMult(gradient,errorsO,0);
    gradient = Matrix.nMult(nn.learningrate,gradient);
    Matrix hiddenLastT = Matrix.transpose(nn.H[nn.H.length - 1]);
	Matrix deltaWeightsHO = Matrix.mMult(gradient, hiddenLastT,0);
    nn.HO = Matrix.mAdd(nn.HO, deltaWeightsHO);
    

    //Adjust weights hidden
    for (int n = nn.H.length - 1; n >= 1; n--) {
        gradient = new Matrix(nn.H[n].m, nn.H[n].n);
        gradient = Matrix.derive(gradient, nn.activationindex);
        gradient = Matrix.mMult(gradient,errorsH[n],0);
		gradient = Matrix.nMult(nn.learningrate,gradient);
		nn.HH[n - 1] = Matrix.mAdd(nn.HH[n - 1], Matrix.mMult(gradient, Matrix.transpose(nn.H[n - 1]),0));
    }

    //Adjust weights inp
    gradient = new Matrix(nn.H[0].m, nn.H[0].n);
    gradient = Matrix.derive((gradient), nn.activationindex);
    gradient = Matrix.mMult(gradient,Matrix.transpose(errorsH[0]),0);
	gradient = Matrix.nMult(nn.learningrate,gradient);

	// Adjust weights
	nn.IH = Matrix.mAdd(nn.IH, Matrix.mMult(gradient, Matrix.transpose(nn.I),0));
    return nn;
  }
}
