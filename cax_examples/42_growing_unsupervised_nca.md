# Growing Unsupervised Neural Cellular Automata [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/42_growing_unsupervised_nca.ipynb)

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
from cax.core.update import NCAUpdate
from cax.nn.pool import Pool
from cax.nn.vae import Encoder
from cax.utils import clip_and_uint8
```

## Configuration


```python
seed = 0

spatial_dims = (28, 28)
features = (1, 32, 32)
latent_size = 8

channel_size = 32
num_kernels = 3
hidden_size = 256
cell_dropout_rate = 0.5

num_steps = 64
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
y_train = jnp.array([y.resize(spatial_dims) for y, _ in ds_train])[..., None] / 255
y_test = jnp.array([y.resize(spatial_dims) for y, _ in ds_test])[..., None] / 255

# Visualize
mediapy.show_images(y_train[:8], width=128, height=128)
```

## Instantiate system


```python
class GrowingUnsupervisedNCA(ComplexSystem):
	"""Unsupervised Neural Cellular Automata class."""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize Unsupervised NCA.

		Args:
			rngs: rng key.

		"""
		self.perceive = ConvPerceive(
			channel_size=channel_size,
			perception_size=num_kernels * channel_size,
			feature_group_count=channel_size,
			rngs=rngs,
		)
		self.update = NCAUpdate(
			channel_size=channel_size,
			perception_size=latent_size + num_kernels * channel_size,
			hidden_layer_sizes=(hidden_size,),
			cell_dropout_rate=cell_dropout_rate,
			zeros_init=True,
			rngs=rngs,
		)
		self.encoder = Encoder(
			spatial_dims=spatial_dims,
			features=features,
			latent_size=latent_size,
			rngs=rngs,
		)

		# Initialize kernel with sobel filters
		kernel = jnp.concatenate([identity_kernel(ndim=2), grad_kernel(ndim=2)], axis=-1)
		kernel = jnp.expand_dims(jnp.concatenate([kernel] * channel_size, axis=-1), axis=-2)
		self.perceive.conv.kernel.value = kernel

	def encode(self, x):
		"""Encode image into latent space."""
		mean, logvar = self.encoder(x)
		return self.encoder.reparameterize(mean, logvar)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		# Broadcast the input vector to match the state shape
		input_shape = (*state.shape[:-1], input.shape[-1])
		input = jnp.broadcast_to(input[..., None, None, :], input_shape)

		# Step
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@nnx.jit
	def render(self, state):
		"""Render state to RGB."""
		gray = state[..., -1:]
		rgb = jnp.repeat(gray, 3, axis=-1)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
```


```python
cs = GrowingUnsupervisedNCA(rngs=rngs)
```


```python
params = nnx.state(cs, nnx.Param)
print("Number of params:", sum(x.size for x in jax.tree.leaves(params)))
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with a single alive cell."""
	# Init state
	state = jnp.zeros(spatial_dims + (channel_size,))
	mid = tuple(size // 2 for size in spatial_dims)

	# Set the center cell to alive
	state = state.at[mid[0], mid[1], -1].set(1.0)

	# Sample a random target y
	y_idx = jax.random.choice(key, y_train.shape[0])
	return state, y_idx
```

## Train

### Pool


```python
key, subkey = jax.random.split(key)

keys = jax.random.split(subkey, pool_size)
state, y_idx = jax.vmap(lambda key: sample_state(key))(keys)

pool = Pool.create({"state": state, "y_idx": y_idx})
```

### Optimizer


```python
lr_sched = optax.linear_schedule(
	init_value=learning_rate, end_value=0.1 * learning_rate, transition_steps=50_000
)

optimizer = optax.chain(
	optax.clip_by_global_norm(1.0),
	optax.adam(learning_rate=lr_sched),
)

grad_params = nnx.All(nnx.Param, nnx.Any(nnx.PathContains("update"), nnx.PathContains("encoder")))
optimizer = nnx.Optimizer(cs, optimizer, wrt=grad_params)
```

### Loss


```python
def mse(state, y):
	"""Mean Squared Error."""
	return jnp.mean(jnp.square(state[..., -1:] - y))
```


```python
@nnx.jit
def loss_fn(cs, state, y, key):
	"""Loss function."""
	z = cs.encode(y)

	state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
	nnx.split_rngs(splits=batch_size)(
		nnx.vmap(
			lambda cs, state, z: cs(state, z, num_steps=num_steps, sow=True),
			in_axes=(state_axes, 0, 0),
		)
	)(cs, state, z)

	# Get intermediate states
	intermediates = nnx.pop(cs, nnx.Intermediate)
	state = intermediates.state.value[0]

	idx = jax.random.randint(key, (batch_size,), num_steps // 2, num_steps)
	state = state[jnp.arange(batch_size), idx]

	loss = mse(state, y)
	return loss, state
```

### Train step


```python
@nnx.jit
def train_step(cs, optimizer, pool, key):
	"""Train step."""
	sample_key, sample_state_key, loss_key = jax.random.split(key, 3)

	# Sample from pool
	pool_idx, batch = pool.sample(sample_key, batch_size=batch_size)
	current_state = batch["state"]
	current_y_idx = batch["y_idx"]
	current_y = y_train[current_y_idx]

	# Sort by descending loss
	sort_idx = jnp.argsort(jax.vmap(mse)(current_state, current_y), descending=True)
	pool_idx = pool_idx[sort_idx]
	current_state = current_state[sort_idx]
	current_y_idx = current_y_idx[sort_idx]

	# Sample a new state to replace the worst
	new_state, new_y_idx = sample_state(sample_state_key)
	current_state = current_state.at[0].set(new_state)
	current_y_idx = current_y_idx.at[0].set(new_y_idx)
	current_y = y_train[current_y_idx]

	(loss, current_state), grad = nnx.value_and_grad(
		loss_fn, has_aux=True, argnums=nnx.DiffState(0, grad_params)
	)(cs, current_state, current_y, loss_key)
	optimizer.update(cs, grad)

	pool = pool.update(pool_idx, {"state": current_state, "y_idx": current_y_idx})
	return loss, pool
```

### Main loop


```python
num_train_steps = 8_192
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
state_init, y_idx = jax.vmap(sample_state)(keys)

y = y_train[y_idx]
z = cs.encode(y)

state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
state_final = nnx.split_rngs(splits=num_examples)(
	nnx.vmap(
		lambda cs, state, z: cs(state, z, num_steps=2 * num_steps, sow=True),
		in_axes=(state_axes, 0, 0),
	)
)(cs, state_init, z)
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

mediapy.show_images(y, width=128, height=128)
mediapy.show_videos(frames, width=128, height=128, codec="gif")
```

## Interpolation


```python
# Sample two random images
key, subkey = jax.random.split(key)
y_idx = jax.random.choice(subkey, y_train.shape[0], shape=(2,))
y = y_train[y_idx]

# Compute latent encodings
z = cs.encode(y)

# Interpolate between the two latent encodings
num_interpolations = 8
alphas = jnp.linspace(0.0, 1.0, num_interpolations)
z = jnp.array([alpha * z[0] + (1 - alpha) * z[1] for alpha in alphas])

# Sample initial state
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, num_interpolations)
state_init, _ = jax.vmap(sample_state)(keys)

# Run
state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
state_final = nnx.split_rngs(splits=num_interpolations)(
	nnx.vmap(
		lambda cs, state, z: cs(state, z, num_steps=num_steps),
		in_axes=(state_axes, 0, 0),
	)
)(cs, state_init, z)
```


```python
frames = nnx.vmap(
	lambda cs, state: cs.render(state),
	in_axes=(None, 0),
)(cs, state_final)

mediapy.show_images(frames, width=128, height=128)
```
