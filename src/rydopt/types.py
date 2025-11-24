from typing import TypeAlias
from numpy.typing import ArrayLike

ParamsTuple: TypeAlias = tuple[float, ArrayLike, ArrayLike, ArrayLike]
FixedParamsTuple: TypeAlias = tuple[bool, ArrayLike, ArrayLike, ArrayLike]
