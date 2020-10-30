# CSR Traversal vs HASH JOIN Challenge

## Description
The goal of this challenge consists in implementing what you learned about CSR and Hash-Join,
 and implement a simple query execution engine able to perform a set of predefined simple queries.

## Dataset
You will work on a dataset provided by the [Stanford Network Analysis Project](https://snap.stanford.edu/index.html).
In detail, you will analyze data coming from the [POCEK](https://snap.stanford.edu/data/soc-Pokec.html) social network.
You are interested only in the relationships between people in the social network. So, the
fine containing the interesting information for this challenge is the `soc-pokec-relationships.txt`.  

The relationship graph has 1.6 million nodes, and 30 million edges. From the description of this dataset: 

```
Contains friendship relations between users. There is one relation per line. Values in the line are tab separated. For example if a row contains 3 5, this means that user 3 has a friend: user 5.
```

## Challenge
These are the tasks that you should complete in the contest.

### Task 1
The first task consists in allowing the engine to build the CSR by reading information the `soc-pokec-relationships.txt`.
In the same way the engine must be able to load in-memory the data contained in the file in a tabular format (this will help you during the hash join implementation).
So you will end up with multiple representations for your graph: CSR and table format. You are free to build other representations (e.g. CSC) if you think they can be useful. Remember to justify your choices, as each additional representaion will increase the memory footprint of your application.

Your loading function should be able to load a fixed-size subset of the graph (e.g. up to 1000 vertices or 1000 edges): this will make testing simpler, as you can play with a smaller portion of the overall graph.

### Task 2

1. Implement the `traverse` functionality for the engine: given a CSR and a vertex ID, the function returns a list of all the neighbors for that vertex. For `depth = 1` it will return the immediate neighbors, for `depth > 1` you have to perform a more complex traversal (e.g. BFS/DFS)

2. Implement the `join` functionality for the engine:g iven a table and a `user_id1`, the function returns a list of `user_id2` associated to the given one. For `depth = 1`, it corresponds to the SQL query:

```
SELECT r1.user_id2 
FROM relationship r1
WHERE r1.user_id1 = "input_ID"
```

While for `depth > 1` it will require a JOIN like:

```
SELECT r2.user_id2 
FROM relationship r1 JOIN relationship r2 
ON r1.user_id2 = r2.user_id1
WHERE r1.user_id1 = "input_ID" 
```
  
### Task 3
Use the two base operation you implemented in order to perform the following queries:
- `(a)->(b)`: all pairs of nodes connected, where `a!=b`
- `(a)->(b)->(c)`: all triplets of nodes connected, where `a!=b!=c`
- All the other variants up to 11 nodes connected together

Measure the time (and the memory) that one approach takes (the CSR traversal) against the other (the Hash-Join). 

### Task 4
Given the results obtained by the previous task, design and implement an heuristic allowing your engine to switch efficiently between CSR and Hash-Join in order to achieve the best execution time for the execution of the path.

Additional metrics in task 3 and 4 are welcome: we greatly appreciate if you can measure how much of the memory bandwidth you are using, or build a Roofline Model (https://en.wikipedia.org/wiki/Roofline_model) of your implementations. Any other evaluation that allows deeper understanding performance and hardware utilization is welcome!

## Submission
You must submit your solution by December 9th 11.59 PM. 
The submission consists in an email to Alberto Parravicini (`alberto.parravicini at polimi.it`), with CC to Guido Walter Di Donato (`guidowalter.didonato at polimi.it`) and Marco Santambrogio (`marco.santambrogio at polimi.it`),
containing the name of the partecipants, and a link to the `git` repository on which you developed your solution.

The repository must contain:

* The source code
* A README that explains how to execute your solution (we will use a copy the POCEK social graph identical to the one provided to you)
* A report describing your findings in tasks 3 and 4, a description of your heuristic, and any other implementative decision you took that you'd like to share with us

Please note:

* Your code must be buildable with standard tools like Maven.
* Tests must be runnable using a bash or python script.
* Keep in mind: the easier for us to replicate your results, the better it is for you!
* External libraries are allowed, as long as you justify their usage and the *core* of the implementation is written by you. You can use existing CSR/Hash-Join implementations, but only as a performance comparison against your custom implementation.
* The report should be 4 pages long at most, and written in double-column Latex, with font-size 10pt.