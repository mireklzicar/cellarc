# Elementary Cellular Automata [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maxencefaldor/cax/blob/main/examples/10_elementary.ipynb)

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

from cax.cs.elementary import Elementary
```

## Configuration


```python
seed = 0

num_steps = 512
spatial_dims = (1_024,)
wolfram_code_int = 110  # Rule 110

rngs = nnx.Rngs(seed)
```

## Instantiate system


```python
wolfram_code = Elementary.wolfram_code_from_rule_number(wolfram_code_int)
wolfram_code
```


```python
cs = Elementary(wolfram_code=wolfram_code, rngs=rngs)
```

## Sample initial state


```python
def sample_state():
	"""Sample a state with a single active cell."""
	state = jnp.zeros((*spatial_dims, 1))
	return state.at[spatial_dims[0] // 2].set(1.0)
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
frame = cs.render(states)

mediapy.show_image(frame)
```
