#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/model/layers/embedding.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 27th 2025 03:27:47 pm                                              #
# Modified   : Wednesday August 27th 2025 03:37:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import numpy as np

# ------------------------------------------------------------------------------------------------ #


class Embedding:
    """A from-scratch embedding layer.

    This layer maps integer token indices to dense vector representations.
    It maintains a weight matrix that is learned during training.

    Attributes:
        weights (np.ndarray): The learnable parameter matrix for the embeddings.
        weights_gradient (np.ndarray): The gradient for the weights.
    """

    def __init__(
        self, vocabulary_size: int, embedding_dimension: int, random_generator
    ):
        self._embedding_dimension = embedding_dimension

        # The learnable parameter matrix for the embeddings.
        self._weights = random_generator.standard_normal(
            (vocabulary_size, embedding_dimension)
        ).astype(np.float32) / (embedding_dimension**0.5)

        # The gradient for the weights, accumulated during the backward pass.
        self._weights_gradient = np.zeros_like(self._weights)

        # A cache for the input indices from the forward pass.
        self._input_indices_cache = None

    @property
    def weights(self) -> np.ndarray:
        """The learnable weight matrix of the embedding layer."""
        return self._weights

    @property
    def weights_gradient(self) -> np.ndarray:
        """The accumulated gradient for the weight matrix."""
        return self._weights_gradient

    def forward(self, token_indices: np.ndarray) -> np.ndarray:
        """Performs the embedding lookup and scales the output.

        Args:
            token_indices: An array of integer token IDs of shape
                (batch_size, sequence_length).

        Returns:
            The corresponding embedding vectors of shape
            (batch_size, sequence_length, embedding_dimension),
            scaled by sqrt(embedding_dimension).
        """
        self._input_indices_cache = token_indices

        # Look up the embeddings from the weight matrix.
        embeddings = self._weights[token_indices]

        # Scale the embeddings as per the "Attention Is All You Need" paper.
        return embeddings * (self._embedding_dimension**0.5)

    def backward(self, output_gradient: np.ndarray) -> None:
        """Calculates the gradient for the embedding matrix.

        Args:
            output_gradient: The gradient of the loss with respect
                to the output of this layer's forward pass. Shape is
                (batch_size, sequence_length, embedding_dimension).
        """
        # The forward pass included a scaling factor. We must account for it
        # in the backward pass using the chain rule.
        gradient_for_weights = output_gradient * (self._embedding_dimension**0.5)

        # Add the gradients to the weight_gradient matrix at the specific
        # indices that were used in the forward pass.
        np.add.at(
            self._weights_gradient, self._input_indices_cache, gradient_for_weights
        )

        # The gradient flow stops here as the input was integers.
        return None

    def zero_grad(self) -> None:
        """Resets the accumulated gradients to zero."""
        self._weights_gradient.fill(0.0)
