# visualization_interface.py

import matplotlib.pyplot as plt
import torch
from PIL import Image
import torchvision.transforms as transforms
from neural_network import Net  # Assuming neural_network.py contains the neural network implementation

# Load the trained model
net = Net()
net.load_state_dict(torch.load("mnist_model.pth"))
net.eval()

# Define the transformation to preprocess input images
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Function to classify digit images
def classify_digit(image):
    # Perform prediction
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Function to visualize weights of the first layer
def visualize_weights():
    weights = net.conv1.weight.data.cpu().numpy()
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('Visualization of Weights in the First Convolutional Layer')
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(weights[i * 4 + j][0], cmap='gray')
            axs[i, j].axis('off')
    plt.show()

# Function to visualize the prediction of custom digit images
def visualize_custom_prediction(image_path):
    # Load and preprocess the input image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    predicted_label = classify_digit(image)
    
    # Display the input image and the predicted label
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'Predicted Label: {predicted_label}')
    plt.show()

# Main menu
def main_menu():
    print("MNIST Visualization Interface")
    print("1. Visualize Weights of the First Layer")
    print("2. Visualize Prediction of Custom Digit Image")
    print("3. Exit")

# Interactive interface loop
while True:
    main_menu()
    choice = input("Enter your choice (1/2/3): ")
    
    if choice == '1':
        visualize_weights()
    elif choice == '2':
        image_path = input("Enter the path to the custom digit image: ")
        visualize_custom_prediction(image_path)
    elif choice == '3':
        print("Exiting the visualization interface...")
        break
    else:
        print("Invalid choice. Please enter a valid option.")
