

-Deep Learning Project-
--- Rob Johnson, Christine Huffman, Rouselene Larson ---

This is the repository for the term project of the authors for the Deep Learning Theory class of Spring 2020. 
The data referenced can be found at https://dmlab.cs.gsu.edu/solar/data/data-comp-2020/
An explanation of the data is found at https://www.kaggle.com/c/bigdata2020-flare-prediction

This project falls into three sections:

* K means
* naive linear model
* two linear models stacked in a trench coat, somewhat like a CNN



-- KNN --

This is the standard lazy model - find the K nearest neighbors, and have them vote on which class the input is.

-- naive linear model --

This is a simple linear model that reshapes our input into a single vector and then uses several hidden layers to make predictions.

-- two linear models stacked in a trench coat --

While the title yields a descriptive image, this is really 33 input linear model layers that output one number, which is then placed in a vector and another linear model placed to make a prediction on this vector. 
