package data;

import java.util.ArrayList;

public interface CompressedSparseRow {

  public void buildFromFile(String filepath);

  public ArrayList<Integer> getNeighbors(Integer vertex_id);

  /* Feel free to add any other useful method you can think of */
}
