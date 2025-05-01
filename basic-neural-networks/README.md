## Neural Network Fundamentals

### Introduction

This folder contains experiments focused on neural network fundamentals. It covers concepts like individual neurons, perceptrons, and simple problem applications. The goal is to build a deeper understanding of neural networks by implementing and visualizing their behavior on basic datasets.

Additional experiments incorporating more complex architectures and learning algorithms will be added in the future.

### Files and Descriptions

1. **[`McCulloch_Pitts_Neuron.ipynb`](./McCulloch_Pitts_Neuron.ipynb)**
   - Introduced the McCulloch-Pitts (MCP) neuron.
   - Demonstrated its operation with an AND gate.
   - Showed how MCP doesn't learn but uses fixed weights and bias.

2. **[`Perceptron.ipynb`](./Perceptron.ipynb)**
   - Built a Perceptron class that learns from data.
   - Trained the perceptron on AND gate data.
   - Visualized decision boundaries to show how the perceptron learns.

3. **[`Perceptron_XOR_fail.ipynb`](./Perceptron_XOR_fail.ipynb)**
   - Trained perceptron on XOR data.
   - Showed failure due to non-linear separability.
   - Explained why perceptron cannot solve XOR.

4. **[`Sigmoid_Neuron.ipynb`](./Sigmoid_Neuron.ipynb)**
   - Built a `SigmoidNeuron` class with manual forward pass.
   - Passed input through weighted sum and sigmoid function.
   - Plotted sigmoid curve to visualize activation behavior.

5. **[`05_sigmoid_neuron_training.py`](./05_sigmoid_neuron_training.py)**
   - Implemented gradient descent for training a sigmoid neuron.
   - Manually computed Mean Squared Error (MSE) loss.
   - Computed gradients manually and updated weights using gradient descent.
   - Trained the model on a simple dataset to demonstrate the learning process.

6. **[`06_simple_mlp.py`](./06_simple_mlp.py)**
   - Built a simple Multi-Layer Perceptron (MLP) with an input layer, hidden layer (sigmoid activation), and output layer (sigmoid activation).
   - Trained the MLP to successfully solve the XOR problem.
   - Visualized the decision boundary (optional).

7. **[`07_tanh_neuron.py`](./07_tanh_neuron.py)**
   - Defined the tanh activation function and its derivative.
   - Built a `TanhNeuron` class with a forward pass.
   - Compared `tanh` vs `sigmoid` visually by plotting their activation curves.
   - Showed the differences in behavior between the two activation functions.

8. **[`08_relu_neuron.py`](./08_relu_neuron.py)**
   - Defined the ReLU activation function and its derivative.
   - Built a `ReLUNeuron` class with a forward pass.
   - Demonstrated the activation of ReLU on both positive and negative inputs.
   - Plotted the ReLU activation curve to show its sparsity behavior (outputs zero for negative inputs).

9. **[`09_softmax_output_layer.py`](./09_softmax_output_layer.py)**
   - Implemented softmax activation for the output layer in a multi-class classification scenario.
   - Converted logits into probabilities using the softmax function.
   - Incorporated categorical cross-entropy loss for training.
   - Validated the model on a simple 3-class classification example.

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab

To install the required dependencies, you can use the following command:
```bash
pip install numpy matplotlib
