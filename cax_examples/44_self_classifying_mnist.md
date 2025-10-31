# Self-classifying MNIST Digits [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/44_self_classifying_mnist.ipynb)

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
from cax.core.update import NCAUpdate
from cax.nn.pool import Pool
from cax.utils import clip_and_uint8
```

## Configuration


```python
seed = 0

spatial_dims = (28, 28)
channel_size = 20
perception_size = 80
hidden_layers_sizes = (80,)
cell_dropout_rate = 0.5

num_steps = 20
pool_size = 1_024
batch_size = 16
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
x_train = jnp.array([x.resize(spatial_dims) for x, _ in ds_train])[..., None] / 255
x_test = jnp.array([x.resize(spatial_dims) for x, _ in ds_test])[..., None] / 255

y_integer_train = jnp.array([y for _, y in ds_train], dtype=jnp.int32)
y_integer_test = jnp.array([y for _, y in ds_test], dtype=jnp.int32)

# Visualize
mediapy.show_images(x_train[:8], width=128, height=128)
```


```python
# fmt: off
color_lookup = jnp.array(
	[
		[128, 0, 0],      # Digit 0
		[230, 25, 75],    # Digit 1
		[70, 240, 240],   # Digit 2
		[210, 245, 60],   # Digit 3
		[250, 190, 190],  # Digit 4
		[170, 110, 40],   # Digit 5
		[170, 255, 195],  # Digit 6
		[165, 163, 159],  # Digit 7
		[0, 128, 128],    # Digit 8
		[128, 128, 0],    # Digit 9
		[0, 0, 0],        # Default
		[255, 255, 255],  # Background
	]
) / 255


def compute_y(x, y_integer):
	"""Compute the target y from image and integer label."""
	mask = x >= 0.1
	return jnp.where(mask, jax.nn.one_hot(y_integer, 10), 0.0)


def render(x, y):
	"""Render x and y to RGB."""
	# Mask for digit and background pixels
	is_digit = (x > 0.1).astype(jnp.float32)
	is_not_digit = 1.0 - is_digit

	# Apply the mask to the probabilities
	y = is_digit * y

	black_and_white = jnp.concatenate([is_digit, is_not_digit], axis=-1) * 0.01
	y = jnp.concatenate([y, black_and_white], axis=-1)

	return color_lookup[jnp.argmax(y, axis=-1)]
```


```python
y_train = jax.vmap(compute_y)(x_train, y_integer_train)
y_test = jax.vmap(compute_y)(x_test, y_integer_test)
```


```python
# Visualize different colored digits
digits = []
for i in range(10):
	mask = y_integer_train == i
	idx = jnp.argmax(mask)
	digits.append(render(x_train[idx], y_train[idx]))

mediapy.show_images(digits, width=64, height=64)
```

## Instantiate system


```python
class SelfClassifyingNCA(ComplexSystem):
	"""Self-Classifying Neural Cellular Automata."""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize Self-Classifying NCA.

		Args:
			rngs: rng key.

		"""
		self.perceive = ConvPerceive(
			channel_size=channel_size,
			perception_size=perception_size,
			use_bias=True,
			activation_fn=nnx.relu,
			rngs=rngs,
		)
		self.update = NCAUpdate(
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=hidden_layers_sizes,
			cell_dropout_rate=cell_dropout_rate,
			zeros_init=True,
			rngs=rngs,
		)

	def _step(self, state: State, input: Input | None = None, *, sow: bool = False) -> State:
		"""Perform a single step."""
		# Extract x
		x = state[..., -1:]

		# Step
		perception = self.perceive(state)
		next_state = self.update(state, perception, input)

		# Override
		next_state = next_state.at[..., -1:].set(x)

		if sow:
			self.sow(nnx.Intermediate, "state", next_state)

		return next_state

	@nnx.jit
	def render(self, state: State):
		"""Render state to RGB frame."""
		# Extract x and classification logits
		x = state[..., -1:]
		logits = state[..., :10]

		# Render the image and the logits to RGB
		rgb = render(x, logits)

		# Clip values to valid range and convert to uint8
		return clip_and_uint8(rgb)
```


```python
cs = SelfClassifyingNCA(rngs=rngs)
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
	state = jnp.zeros(x_train.shape[1:3] + (channel_size,))

	# Sample random image
	x_idx = jax.random.choice(key, x_train.shape[0])
	x = x_train[x_idx]

	# Set image in state
	state = state.at[..., -1:].set(x)
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
	init_value=learning_rate, end_value=0.01 * learning_rate, transition_steps=100_000
)

optimizer = optax.chain(
	optax.clip_by_global_norm(1.0),
	optax.adam(learning_rate=lr_sched),
)

optimizer = nnx.Optimizer(cs, optimizer, wrt=nnx.Param)
```

### Loss


```python
def l2(state, y):
	"""L2."""
	l2_loss = jnp.sum(jnp.square(state[..., :10] - y), axis=(-1, -2, -3)) / 2
	return jnp.mean(l2_loss)


def ce(state, y):
	"""Cross-entropy."""
	integer_label = jnp.argmax(y, axis=-1)
	return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(state[..., :10], integer_label))
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

	loss = l2(state, y)
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

	# A quarter of the batch is replaced with new images
	new_state, new_x_idx = sample_state(sample_state_key)
	current_state = current_state.at[: batch_size // 4].set(new_state)
	current_x_idx = current_x_idx.at[: batch_size // 4].set(new_x_idx)

	# Get images
	current_y = y_train[current_x_idx]

	(loss, current_state), grad = nnx.value_and_grad(loss_fn, has_aux=True)(
		cs, current_state, current_y
	)
	optimizer.update(cs, grad)

	pool = pool.update(pool_idx, {"state": current_state, "x_idx": current_x_idx})
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
state_init, _ = jax.vmap(sample_state)(keys)

state_axes = nnx.StateAxes({nnx.RngState: 0, nnx.Intermediate: 0, ...: None})
state_final = nnx.split_rngs(splits=num_examples)(
	nnx.vmap(
		lambda cs, state: cs(state, num_steps=4 * num_steps, sow=True),
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

mediapy.show_videos(frames, width=128, height=128, codec="gif")
```
