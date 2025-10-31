# Variational Autoencoder [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/50_vae.ipynb)

## Installation

You will need Python 3.11 or later, and a working JAX installation. For example, you can install JAX with:


```python
%pip install -U "jax[cuda]"
```

Then, install CAX from PyPi:


```python
%pip install -U "cax[examples]"
```

## Import


```python
import jax
import jax.numpy as jnp
import mediapy
import optax
import torchvision
from flax import nnx
from tqdm.auto import tqdm

from cax.nn.vae import VAE, vae_loss
```

## Configuration


```python
seed = 0

spatial_dims = (28, 28)
features = (1, 32, 32)
latent_size = 4

batch_size = 32
learning_rate = 1e-2

key = jax.random.key(seed)
rngs = nnx.Rngs(seed)
```

## Dataset


```python
# Load MNIST dataset
ds_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
ds_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)

# Convert to jax.Array
x_train = jnp.array([y.resize(spatial_dims) for y, _ in ds_train])[..., None] / 255
x_test = jnp.array([y.resize(spatial_dims) for y, _ in ds_test])[..., None] / 255

# Visualize
mediapy.show_images(x_train[:8], width=128, height=128)
```

## Instantiate system


```python
vae = VAE(
	spatial_dims=spatial_dims,
	features=features,
	latent_size=latent_size,
	rngs=rngs,
)
```


```python
params = nnx.state(vae, nnx.Param)
print("Number of params:", sum(x.size for x in jax.tree.leaves(params)))
```

## Train

### Optimizer


```python
lr_sched = optax.linear_schedule(
	init_value=learning_rate, end_value=0.01 * learning_rate, transition_steps=8_192
)

optimizer = optax.chain(
	optax.clip_by_global_norm(1.0),
	optax.adam(learning_rate=lr_sched),
)

optimizer = nnx.Optimizer(vae, optimizer, wrt=nnx.Param)
```

### Loss


```python
@nnx.jit
def loss_fn(vae, image):
	"""Loss function."""
	image_recon, mean, logvar = vae(image)
	return vae_loss(image_recon, image, mean, logvar)
```

### Train step


```python
@nnx.jit
def train_step(vae, optimizer, key):
	"""Train step."""
	image_idx = jax.random.choice(key, x_train.shape[0], shape=(batch_size,))
	image = x_train[image_idx]

	loss, grad = nnx.value_and_grad(loss_fn)(vae, image)
	optimizer.update(vae, grad)

	return loss
```

### Main loop


```python
num_train_steps = 8_192
print_interval = 128

pbar = tqdm(range(num_train_steps), desc="Training", unit="train_step")
losses = []
for i in pbar:
	key, subkey = jax.random.split(key)
	loss = train_step(vae, optimizer, subkey)
	losses.append(loss)

	if i % print_interval == 0 or i == num_train_steps - 1:
		avg_loss = sum(losses[-print_interval:]) / len(losses[-print_interval:])
		pbar.set_postfix({"Average Loss": f"{avg_loss:.3e}"})
```

## Visualize


```python
num_examples = 8

key, subkey = jax.random.split(key)
z = jax.random.normal(subkey, shape=(num_examples, latent_size))
x = vae.generate(z)

mediapy.show_images(x, width=128, height=128)
```


```python
num_examples = 8

key, subkey = jax.random.split(key)
x_idx = jax.random.choice(subkey, x_test.shape[0], shape=(num_examples,))
x = x_test[x_idx]

state_axes = nnx.StateAxes({nnx.RngState: 0, ...: None})
x_recon, _, _ = nnx.split_rngs(splits=num_examples)(
	nnx.vmap(
		lambda vae, x: vae(x),
		in_axes=(state_axes, 0),
	)
)(vae, x)

mediapy.show_images(x, width=128, height=128)
mediapy.show_images(jax.nn.sigmoid(x_recon), width=128, height=128)
```
