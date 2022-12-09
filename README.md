# CS5350/6350-Machine-Learning
## This is a machine learning library developed by Mokshagna Sai Teja Karanam for CS5350/6350 in University of Utah.
#### For running the Decision Tree
#### Installation instructions(Run this before you run the other shell-scripts):
```
# Installs pandas, numpy
./install.sh
```  
  
#### Commands to run the shell-scripts
```
# To run various cases question-2a:
# Please use the same strings for metric names: gini, MajorityError, entropy
Examples:
./run_2a.sh 1 gini
./run_2a.sh 3 MajorityError
./run_2a.sh 5 entropy

```

#### For running the Ensemble Learning, Linear Regression

#### Before running the sh files, please make sure permissions are enabled:
#### Installation instructions(Run this before you run the other shell-scripts):
```
# Install pandas, numpy, pickle, matplotlib, xlrd(for bonus) version > 1.0.0
./install.sh
```  

#### For running the Perceptron
#### Before running the sh files, please make sure permissions are enabled by executing the following commands:
#### chmod +x install.sh
#### chmod +x run_2a.sh
#### chmod +x run_2b.sh
#### chmod +x run_2c.sh
#### Installation instructions(Run this before you run the other shell-scripts):
```
# Install pandas, numpy by running
./install.sh
```  
#### For running the SVM
#### Run the sh files for each problem like run_2a.sh and so on:
#### Installation instructions:
```
# Install pandas, numpy, scipy, warnings by running
./install.sh
```  

#### For running the Neural Network
#### Run the sh files for each problem to get the results:
#### please make sure permissions are enabled for installation
#### Installation instructions:
```
# Install pandas, numpy, tensorflow, warnings by running
./install.sh
```  

##### I have created 2 notebooks for 2a question where 2a1 includes running one training example of bank dataset and it will list out all the gradients of weights and it will take 2 arguments represents width of each layer as this question is asking 2 layers give 2 arguments as follows to see the results:
```
run_2a1.sh 5 5

run_2a1.sh 2 4
```
#####For running 2a2 (verify the results of problem 3) run the command:
```
run_2a2.sh
```
