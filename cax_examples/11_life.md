# Conway's Game of Life [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/11_life.ipynb)

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
import jax.numpy as jnp
import mediapy
from flax import nnx

from cax.cs.life import Life
```

## Configuration


```python
seed = 0

num_steps = 128
spatial_dims = (32, 32)
rule_golly = "B3/S23"  # Conway's Game of Life

rngs = nnx.Rngs(seed)
```

## Instantiate system


```python
birth, survival = Life.birth_survival_from_string(rule_golly)
birth, survival
```


```python
cs = Life(birth=birth, survival=survival, rngs=rngs)
```

## Sample initial state


```python
def sample_state():
	"""Sample a state with a glider for the Game of Life."""
	state = jnp.zeros((*spatial_dims, 1))

	mid_x, mid_y = spatial_dims[0] // 2, spatial_dims[1] // 2
	glider = jnp.array(
		[
			[0.0, 1.0, 0.0],
			[0.0, 0.0, 1.0],
			[1.0, 1.0, 1.0],
		]
	)
	return state.at[mid_x : mid_x + 3, mid_y : mid_y + 3, 0].set(glider)
```

## Run


```python
state_init = sample_state()
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
	lambda cs, state: cs.render(state),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_video(frames, width=256, height=256, codec="gif")
```

## Life Family

You can experiment with other [Life-like Cellular Automata](https://en.wikipedia.org/wiki/Life-like_cellular_automaton) by changing the rule.

### [HighLife](https://en.wikipedia.org/wiki/Highlife_(cellular_automaton))


```python
birth, survival = Life.birth_survival_from_string("B36/S23")
cs = Life(birth=birth, survival=survival, rngs=rngs)
```

### [Life without Death](https://en.wikipedia.org/wiki/Life_without_Death)


```python
birth, survival = Life.birth_survival_from_string("B3/S012345678")
cs = Life(birth=birth, survival=survival, rngs=rngs)
```

### Run


```python
state_init = sample_state()
state_final = cs(state_init, num_steps=num_steps, sow=True)
```

### Visualize


```python
intermediates = nnx.pop(cs, nnx.Intermediate)
states = intermediates.state.value[0]
```


```python
states = jnp.concatenate([state_init[None], states])
frames = nnx.vmap(
	lambda cs, state: cs.render(state),
	in_axes=(None, 0),
)(cs, states)

mediapy.show_video(frames, width=256, height=256, codec="gif")
```
