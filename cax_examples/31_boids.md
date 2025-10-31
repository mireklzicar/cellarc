# Boids [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/31_boids.ipynb)

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
from flax import nnx

from cax.cs.boids import BoidPolicy, Boids, BoidsState
```

## Configuration


```python
seed = 0

num_steps = 1024
num_spatial_dims = 2
num_boids = 256
dt = 0.01

acceleration_max = jnp.inf
acceleration_scale = 1.0
perception = 0.1
separation_distance = 0.025

separation_weight = 4.5
alignment_weight = 0.65
cohesion_weight = 0.75
noise_scale = 0.1

key = jax.random.key(seed)
rngs = nnx.Rngs(seed)
```

## Instantiate system


```python
boid_policy = BoidPolicy(
	acceleration_max=acceleration_max,
	acceleration_scale=acceleration_scale,
	perception=perception,
	separation_distance=separation_distance,
	separation_weight=separation_weight,
	alignment_weight=alignment_weight,
	cohesion_weight=cohesion_weight,
	noise_scale=noise_scale,
	rngs=rngs,
)

cs = Boids(
	dt=dt,
	velocity_half_life=jnp.inf,
	boid_policy=boid_policy,
)
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with random positions and velocities."""
	key_position, key_velocity = jax.random.split(key)

	# Position
	position = jax.random.uniform(key_position, (num_boids, num_spatial_dims))

	# Velocity
	velocity = jax.random.uniform(key_velocity, (num_boids, num_spatial_dims))

	return BoidsState(position=position, velocity=velocity)
```

## Run


```python
key, subkey = jax.random.split(key)
state_init = sample_state(subkey)
state_final = cs(state_init, num_steps=num_steps, sow=True)
```

## Visualize


```python
intermediates = nnx.pop(cs, nnx.Intermediate)
states = intermediates.state.value[0]
```


```python
states = jax.tree.map(lambda x, xs: jnp.concatenate([x[None], xs]), state_init, states)
frames = nnx.vmap(
	lambda cs, state: cs.render(state, boids_size=0.01),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_video(frames, width=512, height=512, fps=int(1 / dt))
```

## Boid simulation with custom boid policy


```python
class BoidPolicy(nnx.Module):
	"""Boid policy inspired by the neural network-based reference implementation."""

	def __init__(
		self,
		num_neighbors: int = 16,  # Number of neighbors to consider
		hidden_features: int = 8,  # Hidden layer size from reference
		*,
		acceleration_max: float = jnp.inf,
		acceleration_scale: float = 10.0,  # Scaling factor from reference
		perception: float = 0.1,  # Perception radius
		rngs: nnx.Rngs,
	):
		"""Initialize boid policy."""
		self.num_neighbors = num_neighbors
		self.acceleration_max = acceleration_max
		self.acceleration_scale = acceleration_scale
		self.perception = perception
		self.rngs = rngs

		self.dense1 = nnx.Linear(4, hidden_features, rngs=rngs)
		self.dense2 = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
		self.dense3 = nnx.Linear(hidden_features, hidden_features, rngs=rngs)
		self.dense4 = nnx.Linear(hidden_features, 1, rngs=rngs)

	def _toroidal_vector(self, position_1: jax.Array, position_2: jax.Array) -> jax.Array:
		"""Calculate vector considering toroidal boundaries in [0, 1]^n."""
		pos_diff = position_2 - position_1
		pos_diff = jnp.where(pos_diff > 0.5, pos_diff - 1.0, pos_diff)
		pos_diff = jnp.where(pos_diff < -0.5, pos_diff + 1.0, pos_diff)
		return pos_diff

	def _get_transformation_mats(self, position: jax.Array, velocity: jax.Array):
		"""Compute global-to-local and local-to-global transformation matrices."""
		u, v = velocity / jnp.maximum(jnp.linalg.norm(velocity), 1e-8)  # Normalize velocity
		x, y = position

		# Global to local transformation (including translation)
		global2local = jnp.array([[u, v, -u * x - v * y], [-v, u, v * x - u * y], [0, 0, 1]])

		# Local to global transformation (including translation)
		local2global = jnp.array([[u, -v, x], [v, u, y], [0, 0, 1]])

		# Rotation-only matrices (for velocity)
		global2local_rot = jnp.array([[u, v, 0], [-v, u, 0], [0, 0, 1]])
		local2global_rot = jnp.array([[u, -v, 0], [v, u, 0], [0, 0, 1]])

		return global2local, local2global, global2local_rot, local2global_rot

	def _clip_by_norm(self, vector: jax.Array, max_val: float) -> jax.Array:
		"""Limit the magnitude of a vector."""
		norm = jnp.linalg.norm(vector)
		return jnp.where(norm > max_val, vector * max_val / norm, vector)

	def __call__(self, state: BoidsState, boid_idx: int) -> jax.Array:
		"""Compute acceleration for a boid based on its neighbors.

		Args:
			state: State containing position and velocity of all boids.
			boid_idx: Index of the current boid.

		Returns:
			Acceleration vector for the boid.

		"""
		# Extract current boid's position and velocity
		xi = state.position[boid_idx]
		vi = state.velocity[boid_idx]

		# Compute distances to all other boids
		distances = jax.vmap(lambda pos: jnp.sum(self._toroidal_vector(xi, pos) ** 2))(
			state.position
		)

		# Find nearest neighbors
		idx_neighbor = jnp.argsort(distances)[1 : self.num_neighbors + 1]  # Exclude self
		xn = state.position[idx_neighbor]  # Neighbor positions
		vn = state.velocity[idx_neighbor]  # Neighbor velocities
		neighbor_distances = distances[idx_neighbor]

		# Create mask for neighbors within visual range
		mask = neighbor_distances < self.perception**2

		# Get transformation matrices
		g2l, l2g, g2lr, l2gr = self._get_transformation_mats(xi, vi)

		# Transform neighbor positions to local frame
		xn_hom = jnp.concatenate(
			[xn, jnp.ones((self.num_neighbors, 1))], axis=-1
		)  # Homogeneous coords
		xn_local = jax.vmap(lambda x: g2l @ x)(xn_hom[:, :, None])[:, :2, 0]  # num_neighbors, 2

		# Transform neighbor velocities to local frame (rotation only)
		vn_hom = jnp.concatenate([vn, jnp.ones((self.num_neighbors, 1))], axis=-1)
		vn_local = jax.vmap(lambda v: g2lr @ v)(vn_hom[:, :, None])[:, :2, 0]  # num_neighbors, 2

		# Prepare inputs for the neural network (scale positions as in reference)
		inputs = jnp.concatenate([50.0 * xn_local, vn_local], axis=-1)  # num_neighbors, 4

		# Neural network processing (similar to BoidNetwork)
		x = self.dense1(inputs)  # num_neighbors, hidden_features
		x = nnx.tanh(x)
		x = self.dense2(x)
		x = nnx.tanh(x)

		# Aggregate over neighbors with mask
		x = (x * mask[:, None]).sum(axis=0) / jnp.maximum(mask.sum(), 1e-8)  # hidden_features

		# Final layers
		x = self.dense3(x)
		x = nnx.tanh(x)
		x = self.dense4(x)
		x = nnx.tanh(x)  # Scalar output

		# Handle case with no neighbors
		dv_local = jax.lax.select(
			mask.sum() > 0,
			jnp.array([0.0, x[0]]),  # [x, y] in local frame
			jnp.zeros(2),
		)

		# Scale acceleration
		dv_local = dv_local * self.acceleration_scale

		# Transform back to global frame
		dv_hom = jnp.concatenate([dv_local, jnp.zeros(1)], axis=-1)  # 3D homogeneous
		acceleration = (l2gr @ dv_hom[:, None])[:2, 0]  # Back to 2D global coords

		# Limit acceleration
		acceleration = self._clip_by_norm(acceleration, self.acceleration_max)

		return acceleration
```


```python
boid_policy = BoidPolicy(
	acceleration_scale=2.0,
	rngs=rngs,
)

cs = Boids(
	dt=dt,
	velocity_half_life=jnp.inf,
	boid_policy=boid_policy,
)
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with random positions and velocities."""
	key_position, key_velocity = jax.random.split(key)

	# Position
	position = jax.random.uniform(key_position, (num_boids, num_spatial_dims))

	# Velocity
	velocity = 0.1 * jax.random.uniform(key_velocity, (num_boids, num_spatial_dims))

	return BoidsState(position=position, velocity=velocity)
```

## Run


```python
key, subkey = jax.random.split(key)
state_init = sample_state(subkey)
state_final = cs(state_init, num_steps=num_steps, sow=True)
```

## Visualize


```python
intermediates = nnx.pop(cs, nnx.Intermediate)
states = intermediates.state.value[0]
```


```python
states = jax.tree.map(lambda x, xs: jnp.concatenate([x[None], xs]), state_init, states)
frames = nnx.vmap(
	lambda cs, state: cs.render(state, boids_size=0.01),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_video(frames, width=512, height=512, fps=int(1 / dt))
```
