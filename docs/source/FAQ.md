# Questions

## What is the intuition behind fixed-length memory bank?
Specifically, for `lin` , the intuition behind the structured attention is to replace pooling over the hidden representations with multi-hop attentive representations (fixed length).  What is the benefit for transforming source sequence representations into a fixed length memory bank?

Push the model to be more language-agnostic. Sentence length tends to be language dependent. For example, French tends to produce longer sentences than English. 

Does the attention in attention bridge act as an enhancement of encoder? Will the attention bridge bring any benefits to decoders?
1. If we view attention bridge as a part of encoder, will the overall model be a partially shared encoder (separate lower layers and shared attention bridge) + separate decoders? 

If the shared attention is viewed is a part of encoder for many2one translation and a part of decoder for one2many translation, the shared attention module encoder some language-independent information to enhance encoding or decoding? 

## Models are saved with encoder, decoder, and generator. What is generator?
The generator contains Linear + activation (softmax or sparsesoftmax). 

### Why we need to separately save “generator”?  
It seems unnecessary to separate the generator. Activation functions do not contain trainable parameters.


## What is the difference between `intermediate_output` and `encoder_output`? [🔗](./onmt/attention_bridge.py#L91)

`intermediate_output` is the intermediate output of stacked n-layered attention bridges. `encoder_output` is literally the output of encoder, which was reused in the n-layered `PerceiverAttentionBridgeLayer`.

For `PerceiverAttentionBridgeLayer` where the encoder output is projected into fixed length via `lattent_array`. But why? 

For `PerceiverAttentionBridgeLayer` :

`intermediate_output` and `encoder_output` are used as: 

```python
   S, B, F = encoder_output.shape
   if intermediate_output is not None:
      cross_query = intermediate_output
   else:
      cross_query = self.latent_array.unsqueeze(0).expand(B, -1, -1)
   encoder_output = encoder_output.transpose(0, 1)
```