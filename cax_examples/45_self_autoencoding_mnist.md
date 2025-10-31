# Self-autoencoding MNIST Digits [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/45_self_autoencoding_mnist.ipynb)

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

from cax.core import ComplexSystem, Input, State
from cax.core.perceive import ConvPerceive, grad_kernel, identity_kernel
from cax.core.update.nca_update import NCAUpdate
from cax.nn.pool import Pool
from cax.utils import clip_and_uint8
```

![Self-autoencoding MNIST Digits](../docs/self_autoencoding_mnist.png)

## Configuration


```python
seed = 0

channel_size = 16
spatial_dims = (28, 28, 42)
num_kernels = 4
hidden_size = 256
cell_dropout_rate = 0.5

num_steps = 96
pool_size = 1_024
batch_size = 8
learning_rate = 1e-3

key = jax.random.key(seed)
rngs = nnx.Rngs(seed)
```

## Dataset


```python
# Load MNIST dataset
ds_train = torchvision.datasets.MNIST(root="./data", train=True, download=True)
ds_test = torchvision.datasets.MNIST(root="./data", train=False, download=True)

# Convert to jax.Array
x_train = jnp.array([x.resize(spatial_dims[:2]) for x, _ in ds_train])[..., None] / 255
x_test = jnp.array([x.resize(spatial_dims[:2]) for x, _ in ds_test])[..., None] / 255

# Visualize
mediapy.show_images(x_train[:8], width=128, height=128)
```

## Instantiate system


```python
class SelfAutoencodingNCA(ComplexSystem):
	"""Self-Autoencoding Neural Cellular Automata class."""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize Self-Autoencoding NCA."""
		self.perceive = ConvPerceive(
			channel_size=channel_size,
			perception_size=num_kernels * channel_size,
			kernel_size=(3, 3, 3),
			feature_group_count=channel_size,
			rngs=rngs,
		)
		self.update = NCAUpdate(
			channel_size=channel_size,
			perception_size=num_kernels * channel_size,
			hidden_layer_sizes=(hidden_size,),
			cell_dropout_rate=cell_dropout_rate,
			kernel_size=(3, 3, 3),
			zeros_init=True,
			rngs=rngs,
		)

		# Initialize kernel with sobel filters
		kernel = jnp.concatenate([identity_kernel(ndim=3), grad_kernel(ndim=3)], axis=-1)
		kernel = jnp.expand_dims(jnp.concatenate([kernel] * channel_size, axis=-1), axis=-2)
		self.perceive.conv.kernel.value = kernel

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		"""Perform a single step."""
		# Extract x
		x = state[..., 0, -1:]

		# Step
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		# Mask
		mid = tuple(size // 2 for size in spatial_dims)
		center = next_state[..., *mid, :]
		next_state = next_state.at[..., mid[-1], :].set(0.0)  # Mask
		next_state = next_state.at[..., *mid, :].set(center)  # Except center cell

		# Override
		next_state = next_state.at[..., 0, -1:].set(x)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@nnx.jit
	def render(self, state):
		"""Render state to RGB."""
		gray = state[..., -1, -1:]
		rgb = jnp.repeat(gray, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
```


```python
cs = SelfAutoencodingNCA(rngs=rngs)
```


```python
params = nnx.state(cs, nnx.Param)
print("Number of params:", sum(x.size for x in jax.tree.leaves(params)))
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with a random image."""
	# Init state
	state = jnp.zeros(spatial_dims + (channel_size,))

	# Sample random image
	x_idx = jax.random.choice(key, x_train.shape[0])
	x = x_train[x_idx]

	# Set image in state
	state = state.at[..., 0, -1:].set(x)
	return state, x_idx
```

## Train

### Pool


```python
key, subkey = jax.random.split(key)

keys = jax.random.split(subkey, pool_size)
state, x_idx = jax.vmap(sample_state)(keys)

pool = Pool.create({"state": state, "x_idx": x_idx})
```

### Optimizer


```python
lr_sched = optax.linear_schedule(
	init_value=learning_rate, end_value=0.5 * learning_rate, transition_steps=2_000
)

optimizer = optax.chain(
	optax.clip_by_global_norm(1.0),
	optax.adam(learning_rate=lr_sched),
)

update_params = nnx.All(
	nnx.Param,
	# nnx.PathContains("update"),
)
optimizer = nnx.Optimizer(cs, optimizer, wrt=update_params)
```

### Loss


```python
def mse(state, x):
	"""Mean Squared Error."""
	return jnp.mean(jnp.square(state[..., :, -1:] - x[..., None, :]))
```


```python
@nnx.jit
def loss_fn(cs, state, x):
	"""Loss function."""
	state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
	nnx.split_rngs(splits=batch_size)(
		nnx.vmap(
			lambda cs, state: cs(state, num_steps=num_steps, sow=True),
			in_axes=(state_axes, 0),
		)
	)(cs, state)

	# Get intermediate states
	intermediates = nnx.pop(cs, nnx.Intermediate)
	state = intermediates.state.value[0]

	# Sample a random step
	idx = jax.random.randint(key, (batch_size,), num_steps // 2, num_steps)
	state = state[jnp.arange(batch_size), idx]

	loss = mse(state, x)
	return loss, state
```

### Train step


```python
@nnx.jit
def train_step(cs, optimizer, pool, key):
	"""Train step."""
	sample_key, sample_state_key = jax.random.split(key)

	# Sample from pool
	pool_idx, batch = pool.sample(sample_key, batch_size=batch_size)
	current_state = batch["state"]
	current_x_idx = batch["x_idx"]
	current_x = x_train[current_x_idx]

	# Sort by descending loss
	sort_idx = jnp.argsort(jax.vmap(mse)(current_state, current_x), descending=True)
	pool_idx = pool_idx[sort_idx]
	current_state = current_state[sort_idx]
	current_x_idx = current_x_idx[sort_idx]

	# Sample a new state to replace the worst
	new_state, new_x_idx = sample_state(sample_state_key)
	current_state = current_state.at[0].set(new_state)
	current_x_idx = current_x_idx.at[0].set(new_x_idx)
	current_x = x_train[current_x_idx]

	(loss, current_state), grad = nnx.value_and_grad(
		loss_fn, has_aux=True, argnums=nnx.DiffState(0, update_params)
	)(cs, current_state, current_x)
	optimizer.update(cs, grad)

	pool = pool.update(pool_idx, {"state": current_state, "x_idx": current_x_idx})
	return loss, pool
```

### Main loop


```python
num_train_steps = 2 * 8_192
print_interval = 128

pbar = tqdm(range(num_train_steps), desc="Training", unit="train_step")
losses = []
for i in pbar:
	key, subkey = jax.random.split(key)
	loss, pool = train_step(cs, optimizer, pool, subkey)
	losses.append(loss)

	if i % print_interval == 0 or i == num_train_steps - 1:
		avg_loss = sum(losses[-print_interval:]) / len(losses[-print_interval:])
		pbar.set_postfix({"Average Loss": f"{avg_loss:.3e}"})
```

## Run


```python
num_examples = 8

key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_examples)
state_init, x_idx = jax.vmap(sample_state)(keys)

state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
state_final = nnx.split_rngs(splits=num_examples)(
	nnx.vmap(
		lambda cs, state: cs(state, num_steps=2 * num_steps, sow=True),
		in_axes=(state_axes, 0),
	)
)(cs, state_init)
```

## Visualize


```python
intermediates = nnx.pop(cs, nnx.Intermediate)
states = intermediates.state.value[0]
```


```python
states = jnp.concatenate([state_init[:, None], states], axis=1)
frames = nnx.vmap(
	lambda cs, state: cs.render(state),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_images(x_train[x_idx], width=128, height=128)
mediapy.show_videos(frames, width=128, height=128, codec="gif")
```
