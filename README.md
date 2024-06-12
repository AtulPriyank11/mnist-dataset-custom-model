# mnist-dataset-custom-model

##  Brief Report:
Model Architecture: The neural network architecture used for this task is a convolutional neural network (CNN) designed for image classification tasks. It consists of two convolutional layers followed by max-pooling layers and two fully connected layers. The architecture is as follows:
•	Convolutional Layer 1: Input channels=1 (grayscale images), output channels=32, kernel size=3x3, padding=1 \n
•	Max-Pooling Layer 1: Pool size=2x2, stride=2
•	Convolutional Layer 2: Input channels=32, output channels=64, kernel size=3x3, padding=1
•	Max-Pooling Layer 2: Pool size=2x2, stride=2
•	Fully Connected Layer 1: Input features=6477 (output channels of second convolutional layer), output features=128
•	Fully Connected Layer 2: Input features=128, output features=10 (number of classes in MNIST dataset)
Training Process: The model was trained using the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9). The training process involved the following steps:
•	Data Loading and Preprocessing: The MNIST dataset was loaded and preprocessed using normalization and data augmentation techniques.
•	Model Initialization: A CNN model was initialized with the specified architecture.
•	Loss Function and Optimizer: Cross-entropy loss function and the Adam optimizer were used for training.
•	Training Loop: The model was trained over multiple epochs, with mini-batch gradient descent. Training loss was monitored to assess model performance.
•	Evaluation: After training, the model was evaluated on the test set to calculate accuracy and assess generalization performance.
Results: The trained model achieved an accuracy of approximately 99.08% on the test set, indicating excellent performance in classifying handwritten digits.

## Instructions for Using the Visualization Interface:
To use the visualization interface:
1.	Run the visualization_interface.py script in your Python environment.
2.	Follow the prompts to choose the desired visualization option:
o	Option 1: Visualize weights of the first layer.
o	Option 2: Visualize prediction of a custom digit image.
3.	If selecting option 2:
o	Enter the path to the custom digit image when prompted.
o	The interface will display the image along with the model's prediction for that image.
These instructions provide guidance on how to interact with the visualization interface to explore the model's architecture and predictions visually.

## Deliverables:

1. Model Architecture: - Describe the architecture of the neural network used for MNIST classification. 
2. Training Process: - Explain the training process, including the optimizer, loss function, and training loop.
3. Results: - Discuss the accuracy achieved on the test set and any other relevant metrics. 
4. Instructions on How to Use the Visualisation Interface: - Clear instructions are provided on how to interact with the `visualization_interface.py` script. - Explain the main menu options and their functionalities. 
5. Visualisation: -  [https://link-to-visualization-interface](https://github.com/AtulPriyank11/mnist-dataset-custom-model)
