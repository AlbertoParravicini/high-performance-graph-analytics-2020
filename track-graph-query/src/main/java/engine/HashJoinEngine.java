package engine;

import java.util.ArrayList;

import data.Table;

public interface HashJoinEngine {

  public ArrayList<Integer> join(Table tab1, Integer element_id);

  /* Feel free to add any other useful method you can think of */
}
