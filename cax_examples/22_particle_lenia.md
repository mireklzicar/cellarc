# Particle Lenia [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/22_particle_lenia.ipynb)

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

from cax.cs.particle_lenia import (
	GrowthParams,
	KernelParams,
	ParticleLenia,
	ParticleLeniaRuleParams,
	bell,
)
```

## Configuration


```python
seed = 0

num_steps = 8_192
num_spatial_dims = 2
num_particles = 200
T = 10

key = jax.random.key(seed)
rngs = nnx.Rngs(seed)
```

## Instantiate system

### Rule parameters


```python
mean = 4.0
std = 1.0


def compute_weight(mean, std, num_spatial_dims):
	"""Compute weight for the kernel."""
	r = jnp.linspace(max(mean - 4 * std, 0.0), mean + 4 * std, 51)
	y = bell(r, mean, std) * r ** (num_spatial_dims - 1)
	s = jnp.trapezoid(y, r) * {2: 2, 3: 4}[num_spatial_dims] * jnp.pi
	return 1 / s


weight = compute_weight(mean, std, num_spatial_dims)
```


```python
kernel_params = KernelParams(
	weight=weight,
	mean=mean,
	std=std,
)

growth_params = GrowthParams(
	mean=0.6,
	std=0.15,
)

rule_params = ParticleLeniaRuleParams(
	c_rep=1.0,
	kernel_params=kernel_params,
	growth_params=growth_params,
)
```


```python
cs = ParticleLenia(
	num_spatial_dims=num_spatial_dims,
	T=T,
	rule_params=rule_params,
)
```

## Sample initial state


```python
def sample_state(key):
	"""Sample a state with random particule positions."""
	state = 12.0 * (jax.random.uniform(key, (num_particles, num_spatial_dims)) - 0.5)
	return state
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
states = jnp.concatenate([state_init[None], states])
frames = nnx.vmap(
	lambda cs, state: cs.render(state, resolution=512, particle_radius=0.3),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_video(frames, width=256, height=256, fps=600)
```
