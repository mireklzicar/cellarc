# Recurrent Residual Convolutional Neural Network [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/51_rrcnn.ipynb)

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
from cax.core.perceive import ConvPerceive
from cax.core.update import ResidualUpdate
from cax.utils.render import clip_and_uint8
```

## Configuration


```python
seed = 0

spatial_dims = (28, 28)
channel_size = 64
perception_size = 64
update_hidden_layer_sizes = (128, 128)
cell_dropout_rate = 0.5

num_steps = 64
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
class RRCNN(ComplexSystem):
	"""Recurrent Residual Convolutional Neural Network class."""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize RRCNN.

		Args:
			rngs: rng key.

		"""
		self.perceive = ConvPerceive(
			channel_size=channel_size,
			perception_size=perception_size,
			rngs=rngs,
		)
		self.update = ResidualUpdate(
			num_spatial_dims=len(spatial_dims),
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=update_hidden_layer_sizes,
			cell_dropout_rate=cell_dropout_rate,
			zeros_init=True,
			rngs=rngs,
		)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
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
cs = RRCNN(rngs=rngs)
```


```python
params = nnx.state(cs, nnx.Param)
print("Number of params:", sum(x.size for x in jax.tree.leaves(params)))
```

## Sample initial state


```python
def add_noise(image, alpha, key):
	"""Add noise to the image with a given alpha value."""
	noise = jax.random.normal(key, image.shape)
	noisy_image = (1 - alpha) * image + alpha * noise
	return jnp.clip(noisy_image, 0.0, 1.0)


def sample_state(key):
	"""Sample a state with a randomly sampled image and added noise."""
	state = jnp.zeros(y_train.shape[1:3] + (channel_size,))

	sample_key, alpha_key, noise_key = jax.random.split(key, 3)

	# Sample a target image
	y_idx = jax.random.choice(sample_key, y_train.shape[0])
	y = y_train[y_idx]

	# Add noise
	alpha = jax.random.uniform(alpha_key)
	noisy_y = add_noise(y, alpha, noise_key)

	return state.at[..., -1:].set(noisy_y), y_idx
```

## Train

### Optimizer


```python
lr_sched = optax.linear_schedule(
	init_value=learning_rate, end_value=0.1 * learning_rate, transition_steps=2_000
)

optimizer = optax.chain(
	optax.clip_by_global_norm(1.0),
	optax.adam(learning_rate=lr_sched),
)

update_params = nnx.All(nnx.Param, nnx.PathContains("update"))
optimizer = nnx.Optimizer(cs, optimizer, wrt=update_params)
```

### Loss


```python
def mse(state, y):
	"""Mean Squared Error."""
	return jnp.mean(jnp.square(state[..., -1:] - y))
```


```python
@nnx.jit
def loss_fn(cs, state, y):
	"""Loss function."""
	state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
	state = nnx.split_rngs(splits=batch_size)(
		nnx.vmap(
			lambda cs, state: cs(state, num_steps=num_steps),
			in_axes=(state_axes, 0),
		)
	)(cs, state)
	loss = mse(state, y)
	return loss
```

### Train step


```python
@nnx.jit
def train_step(cs, optimizer, key):
	"""Train step."""
	keys = jax.random.split(key, batch_size)
	current_state, y_idx = jax.vmap(sample_state)(keys)
	y = y_train[y_idx]

	loss, grad = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, update_params))(
		cs, current_state, y
	)
	optimizer.update(cs, grad)

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
	loss = train_step(cs, optimizer, subkey)
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

mediapy.show_images(y_train[y_idx], width=128, height=128)
mediapy.show_videos(frames, width=128, height=128, codec="gif")
```
