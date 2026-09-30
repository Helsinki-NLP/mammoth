# Tips and tricks for training

## Speed up training with token-based minibatches and gradient accumulation/maxibatching

Waiting for synchronization is a common reason for inefficiency in distributed training across multiple devices.
Each device performs its forward and backward pass computation independently,
but there is a synchronization point in communicating the gradients before the optimizer can be stepped.
If just one device is still computing its contribution to the gradient,
all other devices have to wait idly even though they have already finished their work.
There are two approaches to maximize throughput:

1. Balance the workload between devices by using token-based minibatch size `batch_type: tokens`
2. Perform more work before communicating the gradient.
    - Set `accum_count: 10` to accumulate the gradient over 10 minibatches before communicating.
    - Set `lookahead_minibatches: 10` to the same value as `accum_count` to make the dataloader read in one maxibatch at a time,
      and locally sort the contents by length. This minimizes padding waste (only works with `batch_type: tokens`).


## Use `decay_method: linear_warmup` for learning rate scheduling

Select a decay method before tuning the learning rate.
The recommended method is `linear_warmup`,
which ramps up learning rate linearly for `warmup_steps`, then decays it linearly until `train_steps`.


Note that the OpenNMT legacy decay methods have inconsistent scaling of the maximum learning rate.
Changing the decay method also rescales the learning rate in unintuitive ways.

The `rsqrt` method doesn't apply warmup, making it unsuitable for Transformers with SGD or Adam.

Note that `exponential_decay` is not a selectable `decay_method`. It is used as a fallback whenever `start_decay_steps` is set,
and `start_decay_steps` defaults to 50000. With the default `decay_method: none` the learning rate is therefore still multiplied by
`learning_rate_decay` (0.5) every `decay_steps` (10000) steps after step 50000, and this decay does not apply warmup either.


## Don't rely on `max_grad_norm` to save you from too high learning rate

The norm of the gradient of each distributed component is clipped, if it exceeds `max_grad_norm` (1.0 by default).
Don't rely on max_grad_norm to save you from too high learning rate:
as each component is clipped individually (after the gradient sync), renormalization does NOT preserve the direction of the global gradient.


## Recommended minimal opts for the native transformer

You can pass through opts to the training config:

```yaml
heads: 16              # attention heads (model_dim must be divisible by heads)
rotary_pos_emb: true   # rotary positional embeddings (RoPE, currently the only option)
ff_activation: swiglu  # choose between "swiglu" and "gelu", "swiglu" by default
```
Note `RMSNorm` is the only normalisation function used by Mammoth for now. 


## Save storage and speed up config-config by using transforms instead of external preprocessing

There are two approaches to preprocessing (e.g. subword tokenization, prefixing, autoencoder noise, ...)

1. Pre-apply the transforms using an external tool. Write the results to disk. Point Mammoth at the transformed files.
2. Apply the transforms at run time using Mammoth. Point Mammoth at the raw original files.

There are multiple benefits to the latter approach of using Mammoth transforms:

- The transformation is applied online, and the result is not saved to disk. 
  This saves storage, which is especially relevant when using very large corpora and sampling different variations for each minibatch.
- It is easy to apply sampling of different variations for each minibatch, e.g. subword regularization or denoising autoencoder.
- It is easy to use the same corpus files symmetrically (e.g. the same files for English->Finnish and Finnish->English)
  by prefixing the source data with language selection tokens.
