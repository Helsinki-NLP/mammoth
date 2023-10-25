
# Attention Bridge

The embeddings are generated through the self-attention mechanism of the encoder and establish a connection with language-specific decoders that focus their attention on these embeddings. This is why they are referred to as 'bridges'. This architectural element serves to link the encoded information with the decoding process, enhancing the flow of information between different stages of language processing.

There are five types of attention mechanism implemented:

```python
layer_type_to_cls = {
            'lin': LinAttentionBridgeLayer,
            'perceiver': PerceiverAttentionBridgeLayer,
            'simple': SimpleAttentionBridgeLayer,
            'transformer': TransformerAttentionBridgeLayer,
            'feedforward': FeedForwardAttentionBridgeLayer,
        }
```

`intermediate_output` refers to the input of the attention bridge, specifically denoting the output that emerges from the encoder, i.e., \\( \mathbf{H}\in \mathbf{R}^{n\times d_h} \\) `(batch, seq_len, hidden_dim)`. In other words, this is the transformed representation of the input data after it has undergone encoding processes. This intermediate output serves as the foundation for the subsequent attention bridge, facilitating the connection between the encoded information and the subsequent stages of language processing.


## LinAttentionBridgeLayer

The attention bridge employed is based on the structured self-attention mechanism introduced by Lin et al. in their 2017 paper, accessible at [https://arxiv.org/abs/1703.03130](https://arxiv.org/abs/1703.03130). This mechanism is utilized to establish the connection between different components of the architecture.
This attention bridge allows the encoded information to be channeled effectively into subsequent language-specific decoding processes, contributing to the overall performance of the architecture.

## PerceiverAttentionBridgeLayer

The `PerceiverAttentionBridgeLayer` involves a multi-headed dot product self-attention mechanism, where the type of attention (`att_type`) can take on one of two values: 'context' or 'self'. This mechanism is structured as follows:

1. **MultiHeadedAttention**: This module performs multi-headed dot product self-attention. It takes the input data and applies self-attention with multiple heads.

2. **AttentionBridgeNorm**: The output of the multi-headed attention mechanism is passed through a normalization process that helps ensure stable learning..

3. **Linear Layer**: After normalization, the data is fed into a linear layer. This linear transformation can be seen as a learned projection of the attention-weighted data into a new space.

4. **ReLU Activation**: The output of the linear layer undergoes the Rectified Linear Unit (ReLU) activation function.

5. **Linear Layer (Second)**: Another linear layer is applied to the ReLU-activated output.

6. **AttentionBridgeNorm (Second)**: Similar to the earlier normalization step, the output of the second linear layer is normalized using the AttentionBridgeNorm module.

## SimpleAttentionBridgeLayer

The process described involves dot product self-attention. The steps are as follows:

1. **Input Transformation**: Given an input matrix \\(\mathbf{H} \in \mathbb{R}^{d_h \times n}\\), two sets of learned weight matrices are used to transform the input. These weight matrices are \\( \mathbf{W}_1 \in \mathbb{R}^{d_h \times d_a}\\) and \\( \mathbf{W}_2 \in \mathbb{R}^{d_h \times d_a}\\). The multiplication of \\(\mathbf{H}\\) with \\(\mathbf{W}_1\\) and \\( \mathbf{W}_2\\) produces matrices \\( \mathbf{V} \\) and \\( \mathbf{K}\\), respectively:

   - \\( \mathbf{V} = \mathbf{H} \mathbf{W}_1\\)
   - \\( \mathbf{K} = \mathbf{H} \mathbf{W}_2\\)

2. **Attention Calculation**: The core attention calculation involves three matrices: \\( \mathbf{Q} \in \mathbb{R}^{d_h \times n}\\), \\( \mathbf{K} \\) (calculated previously), and \\( \mathbf{V}\\) (calculated previously). The dot product of \\( \mathbf{Q}\\) and \\(\mathbf{K}^\top\\) is divided by the square root of the dimensionality of the input features (\\( \sqrt{d_h}\\)).
The final attended output is calculated by multiplying the attention weights with the \\( \mathbf{V} \\) matrix: \\( \mathbf{H}^\prime = \operatorname{Softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}})\mathbf{V}\\)


## TransformerAttentionBridgeLayer

The TransformerEncoderLayer employs multi-headed dot product self-attention (by  `TransformerEncoderLayer`) to capture relationships within the input sequence.

## FeedForwardAttentionBridgeLayer

The `FeedForwardAttentionBridgeLayer` module applies a sequence of linear transformations and `ReLU` activations to the input data, followed by an attention bridge normalization, enhancing the connectivity between different parts of the model.
