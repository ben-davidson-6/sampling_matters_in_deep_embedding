# Sampling matters in deep embedding learning

Implementation in tensorflow (1.x) of the metric learning loss described in [this paper](https://arxiv.org/abs/1706.07567). An example of using this loss with mnist is also given.

# To use the loss

```python
import margin_loss.margin_loss

beta_0 = 1.2 # initial learnable beta value
number_identities = 10 # for mnist

# parameters for the loss, follows those given in the
# authors pytorch implementation at 
params = {
    'alpha': 0.2,
    'nu': 0.,
    'cutoff': 0.5,
    'add_summary': True,
}


# construct your graph which should output an
# embedding tensor

# these are the 
betas = tf.get_variable(
    'beta_margins', 
    initializer=beta_0*tf.ones([10]))

# labels shape [n] dtype int the identity of embedding
# embedding [n, d] dtype float
# beta [#num_identities] margin for every possible value
# label can take
loss = margin_loss(labels, embedding, beta, params)
```