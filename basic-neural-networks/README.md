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

### Requirements

- Python 3.x
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab

To install the required dependencies, you can use the following command:
```bash
pip install numpy matplotlib
