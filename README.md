# QML-for-Conspicuity-Detection-in-Production
Womanium Quantum+AI 2024 Projects

**Please review the participation guidelines [here](https://github.com/womanium-quantum/Quantum-AI-2024) before starting the project.**

_**Do NOT delete/ edit the format of this read.me file.**_

_**Include all necessary information only as per the given format.**_

## Project Information:

### Team Size:
  - Maximum team size = 2
  - While individual participation is also welcome, we highly recommend team participation :)

### Eligibility:
  - All nationalities, genders, and age groups are welcome to participate in the projects.
  - All team participants must be enrolled in Womanium Quantum+AI 2024.
  - Everyone is eligible to participate in this project and win Womanium grants.
  - All successful project submissions earn the Womanium Project Certificate.
  - Best participants win Womanium QSL fellowships with Fraunhofer ITWM. Please review the eligibility criteria for QSL fellowships in the project description below.

### Project Description:
  - Click [here](https://drive.google.com/file/d/1AcctFeXjchtEhYzPUsHpP_b4HGlI4kq9/view?usp=sharing) to view the project description.
  - YouTube recording of the project description - [link](https://youtu.be/Ac1ihFcTRTc?si=i6AIVfQQh8ymYQYp)

## Project Submission:
1. Task-1-: Familiarizing Myself with Pennylane:
   As I have done the task, I discovered that Pennylane is an open-source, cross-platform framework for quantum computing that automates several operations, offers a user-friendly API, and strives to make quantum computing more approachable and user-friendly for 
   academics and developers. By doing the sections “Introduction to Quantum Computing”, “Single-Qubit Gates” and “Circuits with Many Qubits” from the pennylane codebook, I have come to know that quantum computing is based on two fundamental concepts: qubits (quantum 
   bits), which can exist in multiple states simultaneously due to superposition and entanglement, allowing them to be correlated with other qubits even at large distances. 
   To manipulate these qubits, users apply single-qubit gates like Pauli X, Y, Z, Hadamard, and Phase shift gates, which transform the qubit's state. By combining these gates and using multi-qubit operations such as entangling gates (e.g., CNOT) and measurement/control 
   operations, complex quantum circuits can be created to solve problems that are beyond the reach of classical computers.

2. Task-2-: Variotional Classifier:
   In this task we have 2 different parts. 1st one is a parity function of the variotional classifier. In here, we do a comprehensive implementation of a variational classifier using quantum circuits for binary classification tasks. Our goal in here is to train this 
   model on a given dataset using a specified optimizer, and then evaluate its performance on unseen data.

   Key Components : 


   1. Data Loading: The load_data function loads the training and testing data from text files into NumPy arrays.

   2. Quantum Circuit: The quantum circuit is defined by the circuit function, which takes in weights and an input vector x. It prepares the state using state_preparation, applies a series of operations to generate a classification output, and returns the result.

   3. Variational Classifier: The variational_classifier function wraps the quantum circuit with a sigmoid activation function to produce a binary classification output.

   4. Cost Function: The cost function calculates the mean squared error loss between the true labels and predicted outputs.

   5. Optimization: The train_model function trains the variational classifier using an optimizer (in this case, Nesterov Momentum Optimizer) to minimize the cost function over multiple iterations.

   6. Model Evaluation: The evaluate_model function assesses the trained model's performance on unseen data.


   2nd one is Iris Classification with Quantum-Machine-Learning. Here, we see the application of quantum-machine-learning techniques to classify the Iris dataset. 

   Key Components :


   1. Quantum State Preparation: We utilize a quantum state preparation routine to encode classical data into a quantum state. The encoding scheme is based on established quantum computing principles for efficient data representation.
   
   2. Variational Quantum Classifier: A variational quantum circuit is constructed, comprising parameterized quantum gates and entanglement operations. The circuit takes the prepared quantum state as input and produces an expectation value used for prediction.
   
   3. Classical-Quantum Integration: The output of the quantum circuit is combined with a classical bias term to generate the final classification prediction. This step seamlessly integrates the quantum and classical components of the model.
  
   4. Training and Evaluation: The model is trained using a NesterovMomentumOptimizer to minimize the square loss between predicted and true labels. The performance of the classifier is assessed on both training and validation sets using accuracy metrics.
  
   5. Visualization: The notebook includes visualizations of the original data, transformed data, feature vectors, and decision boundaries. These plots provide insights into the data transformations and the classification performance of the model.

3. Task-3-: Quanvolutional Neural Networks (QNN):

   This notebook has guided me through the foundational concepts and practical implementation of a Quantum Neural Network (QNN) for image classification. I have learned how to build, train, and evaluate a QNN using quantum computing frameworks, as well as 
   how to integrate quantum and classical components in a hybrid model.

   key Components :

   1.Quantum Convolution: A quantum circuit is defined to encode a small portion of the input image using parameterized rotations. A random quantum circuit is then applied, followed by measurements to obtain expectation values. These expectation values are 
     treated as features extracted from the image.

   2. Preprocessing: The entire training and test datasets are preprocessed using this quantum convolution operation. This step essentially embeds quantum-derived features into the image data.
  
   3. Classical Model: A simple classical neural network consists of a flattening layer and a dense layer with softmax activation. This model takes the quantum preprocessed images as input.
  
   4. Training and Comparison: The classical model is trained on both the quantum preprocessed data and the original MNIST data. The performance (validation accuracy and loss) of both models is then compared to assess the impact of the quantum convolution layer.

   
4. Task-4-: Quantum Machine Learning for Sine Function:

   This notebook serves as a thorough exercise in applying a Quantum-Machine-Learning model to the approximation of a sine function. Everything is covered, including configuring the quantum environment, creating and refining the quantum circuit, assessing and 
   displaying the model's output. In addition to providing hands-on experience with Pennylane, this interactive project explains how machine learning tasks might potentially benefit from quantum computing.


   Key Components:
   
   1. Data Generation: We generate training and test datasets consisting of input values within a specified range and their corresponding sine values.
  
   2. Quantum Circuit Design: A quantum circuit is defined using PennyLane, incorporating parameterized quantum gates (RX, RY, RZ) to process input data and produce an output.
  
   3. Cost Function Definition: A cost function (MSE) measures the discrepancy between the model's predictions and the actual sine values.
  
   4. Optimization: The Gradient Descent Optimizer iteratively adjusts the circuit parameters to minimize the cost function.
  
   5. Training and Evaluation: The model is trained and then periodically evaluated using an accuracy function to see how well it performs on the test dataset.
  
   6. Visualization: A scatter plot is used to exhibit the final findings, training and test data, and model predictions, for glances.







 
   
   

### Team Information:
Team Member 1:
 - Full Name: Prokash Chandra Roy
 - Womanium Program Enrollment ID (see Welcome Email, format- WQ24-xxxxxxxxxxxxxxx): WQ24-FxzzKzeceZXhHY6

### Project Solution:
_Include a comprehensive summary of all important information about your project solution here._
All necessary code files and any additional information required to judge your project solution should be included in the repository. 

### Project Presentation Deck:
_Link a 5min. presentation recording or deck here._

