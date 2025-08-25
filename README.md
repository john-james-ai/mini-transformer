# MiniTransformer

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A from-scratch implementation of the Transformer architecture in NumPy.

---

## Overview

**MiniTransformer** is a minimalist, pure-NumPy implementation of the original Transformer model introduced in the paper "Attention Is All You Need" by Vaswani et al. This project was built to demystify the inner workings of the Transformer by stripping away the abstractions of modern deep-learning frameworks such as PyTorch and TensorFlow.

The goal is to provide a clear, concise, and understandable codebase that demonstrates the core mechanics of self-attention, positional encodings, and the encoder-decoder stack.

## ✨ Features

-   **Scaled Dot-Product Attention:** The fundamental building block of the model.
-   **Multi-Head Attention:** Implementation of the mechanism to attend to information from different representation subspaces.
-   **Position-wise Feed-Forward Networks:** The fully connected feed-forward network applied to each position separately.
-   **Sinusoidal Positional Encoding:** The classic method to inject sequence order information.
-   **Encoder & Decoder Stacks:** Full implementation of both the encoder and decoder blocks.
-   **Layer Normalization & Residual Connections:** Key components for stabilizing training in deep networks.
-   **Masking:** Correctly implemented source padding masks and target look-ahead masks.

## 🏛️ Architecture

The model follows the architecture described in the original paper. It consists of an encoder stack and a decoder stack. Each encoder layer has a multi-head self-attention mechanism followed by a position-wise feed-forward network. Each decoder layer includes two multi-head attention mechanisms (one self-attention and one cross-attention over the encoder's output) followed by a feed-forward network.

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+
-   NumPy

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/MiniTransformer.git](https://github.com/your-username/MiniTransformer.git)
    cd MiniTransformer
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The project includes scripts for training the model and performing inference.

### Training

To train a new model on a sample dataset, run the training script. You will need to provide pre-tokenized data in the expected format.

```bash
python train.py --data_path /path/to/your/data --config model_config.json
```

### Inference

To translate a sentence using a pre-trained model, use the inference script.

```bash
python translate.py --model_path /path/to/model.npz --sentence "Hello world"
```

## 📁 Project Structure

```
MiniTransformer/
├── data/                  # Directory for sample data
├── minitransformer/        # Main source code
│   ├── attention.py       # Attention mechanisms
│   ├── layers.py          # Encoder and Decoder layers
│   ├── model.py           # The main Transformer model class
│   ├── modules.py         # FFN, LayerNorm, etc.
│   └── positional.py      # Positional encoding
├── notebooks/             # Jupyter notebooks for exploration
├── scripts/               # Helper scripts for data processing
├── tests/                 # Unit tests for the components
├── train.py               # Script to train the model
├── translate.py           # Script for inference
└── requirements.txt       # Project dependencies
```

## 📝 To-Do

-   [ ] Implement learning rate scheduling (e.g., Adam with warmup and decay).
-   [ ] Add beam search decoding for more robust inference.
-   [ ] Implement dropout for regularization.
-   [ ] Add detailed logging and visualization of attention maps.
-   [ ] Expand unit tests for better coverage.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or suggestions.

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

-   This project is heavily inspired by the original paper: [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
-   "The Annotated Transformer" by Harvard NLP for its excellent line-by-line explanation.
