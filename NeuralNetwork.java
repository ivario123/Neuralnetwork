import java.io.Serializable;
import java.io.*;
import MatrixMath.*;
class NN implements Serializable { 
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
  public NN(int[] arc, int activation) {
    this.activation[activation] = true;
    this.arc = arc;
    I = new Matrix(arc[0], 1);
    IH = new Matrix(arc[1], arc[0]);
    IH.randomize();
    IH=IH.rAdd(IH);
    HH = new Matrix[arc.length-2];
    H = new Matrix[arc.length-2];
    for (int i = 0; i < arc.length-2; i++) {
      H[i] = new Matrix(arc[i+1], 1);
      HH[i] = new Matrix(arc[i+2], arc[i+1]);
      HH[i].randomize();
      HH[i] = HH[i].rAdd(HH[i]);
    }
    HO = new Matrix(arc[arc.length-1], arc[arc.length-2]);
    HO.randomize();
    HO = HO.rAdd(HO);
    O = new Matrix(HO.m, 1);
  }
  static void serialize(NN nn) {
    if (nn.fitness > deserialize().fitness) {
      try {
        
        FileOutputStream fileOut =
        new FileOutputStream("bestnn.nn");
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(nn);
        out.close();
        fileOut.close();
        System.out.printf("Serialized data is saved in bestnn.nn");
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
  static NN deserialize() {
    int[] arc = {7, 4, 3, 2};
    NN nn=null;
    try {
      FileInputStream fileIn = new FileInputStream("bestnn.nn");
      ObjectInputStream in = new ObjectInputStream(fileIn);
      nn = (NN) in.readObject();
      in.close();
      fileIn.close();
    } 
    catch (IOException i) {
      //i.printStackTrace();
      return new NN(arc, 1);
    } 
    catch (ClassNotFoundException c) {
      //c.printStackTrace();
      return new NN(arc, 1);
    }
    //System.out.println("NN loaded"+" It's fitness is "+nn.fitness);
    return nn;
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
    O = O.tanh(O);
    float[] out = new float[O.m];
    for (int i = 0; i < O.m; i++) {
      out[i]= O.table[i][0];
    }
    return out;
  }
}