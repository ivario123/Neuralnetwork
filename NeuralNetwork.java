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
  boolean[] activation = {false, false, false, false, false};//tanh,sin, cos,tanh, sigmoid
  public NeuralNetwork(int[] arc, int activation) {
    this.activation[activation] = true;
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
    for (int i = 0; i < inp.length; i++) {
      I.table[i][0] = inp[i];
    }
    I = Matrix.tanh(I);
    H[0] = Matrix.mMult(IH, I);
    for (int i = 1; i < HH.length; i++) {
      H[i] = Matrix.tanh(H[i]);
      H[i] = Matrix.mMult(HH[i-1], H[i-1]);
    }
    O = Matrix.mMult(HO, H[H.length-1]);
    O = Matrix.tanh(O);
    float[] out = new float[O.m];
    for (int i = 0; i < O.m; i++) {
      out[i]= O.table[i][0];
    }
    return out;
  }
}
