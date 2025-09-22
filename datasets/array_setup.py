import numpy as np

from datasets.datautils import audiowu_high_array_geometry


realman_array = audiowu_high_array_geometry()


REAL_MAN = {
    'array_type': 'planar',
    'mic_pos': realman_array[[1, 2, 3, 4, 5, 6, 7, 8, 0]] 

ARRAY_SETUPS = {
    "realman": REAL_MAN,
}
