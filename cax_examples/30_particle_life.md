# Particle Life [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/30_particle_life.ipynb)

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

from cax.cs.particle_life import ParticleLife, ParticleLifeState
```

## Configuration


```python
seed = 0

num_steps = 1024
num_spatial_dims = 2
num_particles = 4096
num_classes = 6
dt = 0.01
force_factor = 1.0
velocity_half_life = dt
r_max = 0.15
beta = 0.3

key = jax.random.key(seed)
rngs = nnx.Rngs(seed)
```

## Instantiate system


```python
# Sample attraction matrix
key, subkey = jax.random.split(key)
A = jax.random.uniform(subkey, (num_classes, num_classes), minval=-1.0, maxval=1.0)
A
```


```python
cs = ParticleLife(
	num_classes=num_classes,
	dt=dt,
	force_factor=force_factor,
	velocity_half_life=velocity_half_life,
	r_max=r_max,
	beta=beta,
	A=A,
)
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with random classes and positions, and zero velocity."""
	key_class, key_position = jax.random.split(key)

	# Class
	class_ = jax.random.choice(key_class, num_classes, (num_particles,))

	# Position
	position = jax.random.uniform(
		key_position, (num_particles, num_spatial_dims), minval=0.0, maxval=1.0
	)

	# Velocity
	velocity = jnp.zeros((num_particles, num_spatial_dims))

	return ParticleLifeState(class_=class_, position=position, velocity=velocity)
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
	lambda cs, state: cs.render(state, particle_radius=0.003),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_video(frames, width=512, height=512, fps=int(1 / dt))
```
