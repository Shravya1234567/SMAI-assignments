# Assignment-1

## Running the shell script
- The shell script can be run by using the command `./eval.sh <filename>` where filename is the test file used to evaluate.

## Assumptions:
- The input file to the bash script is to be given as a .npy file and it is passed as an argument while running the bash script
- The file test.py is called by the bash script to validate the test data.

## Files
- eval.sh : Bash script that takes the input file as test data and then uses it to evaluate
- test.py : Python script used for evaluation of test data provided. It is called by the bash script eval.sh.
- 1.ipynb : Jupyter Notebook containing the code and graphs.
- data.npy : Dataset for KNN classifier
- advertising.csv : Dataset for decision trees