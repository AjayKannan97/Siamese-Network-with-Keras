# Siamese-Network-with-Keras

In the Siamese Network, I used one such methodology. The model is trained with convolutional neural network with RMSprop as an optimiser to update the parameters of all the layers using back propagation for both the lines.
Two dataset have been worked out.
Triplet Loss -
20 alphabets - train & other 6 test - Vertical Split
Acc - test: 70.42378917378917 (for test set) 
Positive similarity: 0.20598295 - Negative similarity 0.18979926 (for training set)
26 alphabets - train & test - Horizontal Split
Acc - test: 72.06604409357851 (for test set)
Positive similarity: 0.2059852 - Negative similarity 0.189592 (for training set)
Contrastive Loss -
20 alphabets - train & other 6 test - Vertical Split
Acc - test: 95.40242165242165 (for test set)
Positive similarity: 0.20598295 - Negative similarity 0.18979926 (for training set)
26 alphabets - train & test - Horizontal Split
Acc - test: 95.51121594300568 (for test set)
Positive similarity: 0.2059852 - Negative similarity 0.18961547 (for training set)
