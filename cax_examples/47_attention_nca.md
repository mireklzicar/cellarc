# Attention-based Neural Cellular Automata [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/47_attention_nca.ipynb)

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
from cax.core.perceive import Perceive, Perception
from cax.core.update import ResidualUpdate
from cax.nn.pool import Pool
from cax.utils import clip_and_uint8
```

## Configuration


```python
seed = 0

spatial_dims = (28, 28)
channel_size = 32
perception_size = 64
num_heads = 4
hidden_size = 128
proj_size = 32
cell_dropout_rate = 0.5

num_steps = 32
pool_size = 1_024
batch_size = 4
learning_rate = 1e-3
mask_ratio = 0.5

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
class ViTPerceive(Perceive):
	"""Vision Transformer Perceive class."""

	def __init__(
		self,
		channel_size: int,
		perception_size: int,
		*,
		num_heads: int,
		hidden_size: int,
		proj_size: int,
		max_position_size: int,
		position_embed_features: int = 4,
		rngs: nnx.Rngs,
	):
		"""Initialize ViT Perceive."""
		self.linear = nnx.Linear(in_features=channel_size, out_features=proj_size, rngs=rngs)

		self.position_embed_features = position_embed_features
		self.position_embed = nnx.Embed(
			num_embeddings=max_position_size,
			features=position_embed_features,
			rngs=rngs,
		)

		self.attention = nnx.MultiHeadAttention(
			num_heads=num_heads,
			in_features=proj_size + 2 * position_embed_features,
			qkv_features=hidden_size,
			out_features=perception_size,
			decode=False,
			rngs=rngs,
		)

	def __call__(self, state: State) -> Perception:
		"""Apply perception to the input state.

		Args:
			state: State of the cellular automaton.

		Returns:
			The perceived state after applying convolutional layers.

		"""
		# Linear projection of state into tokens
		state = self.linear(state)

		# Concatenate position embed
		position_embed_h = self.position_embed(jnp.arange(state.shape[-3]))
		position_embed_w = self.position_embed(jnp.arange(state.shape[-2]))
		position_embed = jnp.concatenate(
			[
				jnp.repeat(position_embed_h[:, None, :], state.shape[-2], axis=1),
				jnp.repeat(position_embed_w[None, :, :], state.shape[-3], axis=0),
			],
			axis=-1,
		)
		tokens = jnp.concatenate([state, position_embed], axis=-1)

		# Get mask for localized attention
		mask = self.get_mask(tokens)

		# Flatten grid into a sequence of tokens
		tokens = jnp.reshape(tokens, tokens.shape[:-3] + (-1, tokens.shape[-1]))

		# Apply localized attention
		perception = self.attention(tokens, mask=mask)
		perception = jnp.reshape(
			perception,
			perception.shape[:-2] + (state.shape[-3], state.shape[-2], perception.shape[-1]),
		)

		return perception

	def get_mask(self, tokens: jax.Array) -> jax.Array:
		"""Get mask for localized attention using Moore neighborhood.

		Args:
			tokens: Input tokens with shape [..., H, W, C]

		Returns:
			Boolean mask with shape [..., H*W, H*W] where True values indicate
			allowed attention connections between tokens.

		"""
		h, w = tokens.shape[-3], tokens.shape[-2]

		# Create position indices
		row_idx = jnp.arange(h)[:, None, None, None]  # [H, 1, 1, 1]
		col_idx = jnp.arange(w)[None, :, None, None]  # [1, W, 1, 1]

		# Broadcast to full grid
		row1 = jnp.broadcast_to(row_idx, (h, w, h, w))  # Source positions
		col1 = jnp.broadcast_to(col_idx, (h, w, h, w))
		row2 = jnp.broadcast_to(row_idx.transpose((2, 3, 0, 1)), (h, w, h, w))  # Target positions
		col2 = jnp.broadcast_to(col_idx.transpose((2, 3, 0, 1)), (h, w, h, w))

		# Calculate Manhattan distance between all positions
		row_dist = jnp.abs(row1 - row2)
		col_dist = jnp.abs(col1 - col2)

		# Create mask where True allows attention (distance <= 1 in both dimensions)
		mask = (row_dist <= 1) & (col_dist <= 1)

		# Reshape to attention matrix shape
		mask = jnp.reshape(mask, (h * w, h * w))

		return mask
```


```python
class ViTNCA(ComplexSystem):
	"""ViT Neural Cellular Automata class."""

	def __init__(self, *, rngs: nnx.Rngs):
		"""Initialize ViT NCA.

		Args:
			rngs: rng key.

		"""
		self.perceive = ViTPerceive(
			channel_size=channel_size,
			perception_size=perception_size,
			num_heads=num_heads,
			hidden_size=hidden_size,
			proj_size=proj_size,
			max_position_size=max(spatial_dims),
			rngs=rngs,
		)
		self.update = ResidualUpdate(
			num_spatial_dims=2,
			channel_size=channel_size,
			perception_size=perception_size,
			hidden_layer_sizes=(hidden_size,),
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
cs = ViTNCA(rngs=rngs)
```


```python
params = nnx.state(cs, nnx.Param)
print("Number of params:", sum(x.size for x in jax.tree.leaves(params)))
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with a randomly masked image."""
	# Sample image from dataset
	y_idx = jax.random.choice(key, y_train.shape[0])
	y = y_train[y_idx]
	y_channel_size = y.shape[-1]

	# Init state with zeros
	state = jnp.zeros(spatial_dims + (channel_size,))

	# Mask pixels randomly
	mask = jax.random.bernoulli(key, 1 - mask_ratio, shape=spatial_dims)
	mask = jnp.expand_dims(mask, axis=-1)
	y *= mask

	# Set state
	state = state.at[..., -y_channel_size:].set(y)

	return state, y_idx


def sample_state_test(key):
	"""Sample a state with a randomly masked image."""
	# Sample image from dataset
	y_idx = jax.random.choice(key, y_test.shape[0])
	y = y_test[y_idx]
	y_channel_size = y.shape[-1]

	# Init state with zeros
	state = jnp.zeros(spatial_dims + (channel_size,))

	# Mask pixels randomly
	mask = jax.random.bernoulli(key, 1 - mask_ratio, shape=spatial_dims)
	mask = jnp.expand_dims(mask, axis=-1)
	y *= mask

	# Set state
	state = state.at[..., -y_channel_size:].set(y)

	return state, y_idx
```

## Train

### Pool


```python
key, subkey = jax.random.split(key)

keys = jax.random.split(subkey, pool_size)
state, y_idx = jax.vmap(sample_state)(keys)

pool = Pool.create({"state": state, "y_idx": y_idx})
```

### Optimizer


```python
lr_sched = optax.linear_schedule(
	init_value=learning_rate, end_value=0.1 * learning_rate, transition_steps=4_096
)

optimizer = optax.chain(
	optax.clip_by_global_norm(1.0),
	optax.adam(learning_rate=lr_sched),
)

optimizer = nnx.Optimizer(cs, optimizer, wrt=nnx.Param)
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
	current_y_idx = batch["y_idx"]
	current_y = y_train[current_y_idx]

	# Sort by descending loss
	sort_idx = jnp.argsort(jax.vmap(mse)(current_state, current_y), descending=True)
	pool_idx = pool_idx[sort_idx]
	current_state = current_state[sort_idx]
	current_y_idx = current_y_idx[sort_idx]

	# Sample a new image to replace the worst
	new_state, new_y_idx = sample_state(sample_state_key)
	current_state = current_state.at[0].set(new_state)
	current_y_idx = current_y_idx.at[0].set(new_y_idx)
	current_y = y_train[current_y_idx]

	(loss, current_state), grad = nnx.value_and_grad(loss_fn, has_aux=True)(
		cs, current_state, current_y
	)
	optimizer.update(cs, grad)

	pool = pool.update(pool_idx, {"state": current_state, "y_idx": current_y_idx})
	return loss, pool
```

### Main loop


```python
num_train_steps = 8_196
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
state_init, y_idx = jax.vmap(sample_state_test)(keys)

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
frame_init = nnx.vmap(
	lambda cs, state: cs.render(state),
	in_axes=(None, 0),
)(cs, state_init)

frames = nnx.vmap(
	lambda cs, state: cs.render(state),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_images(y_test[y_idx], width=128, height=128)
mediapy.show_images(frame_init, width=128, height=128)
mediapy.show_videos(frames, width=128, height=128, codec="gif")
```
