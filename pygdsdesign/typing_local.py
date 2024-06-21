import numpy as np
from typing import List, Tuple, Union, List

# Contains some typing used in the library
Coordinate = Union[Tuple[Union[float, int], Union[float, int]],
                   List[Union[float, int]]]
Coordinates = List[np.ndarray]

Number = Union[float, int]
