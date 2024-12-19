from typing import Union
from jaxtyping import Array, Float
from jax.random import PRNGKey as _PRNGKey


PRNGKey = _PRNGKey

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
