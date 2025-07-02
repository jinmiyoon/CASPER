import inspect
import os
import sys

import numpy as np

# ✅ Force Python to prioritize the local casper/ path
from casper.interface.segment import Segment

# ✅ Force Python to prioritize the LOCAL 'casper/' over site-packages
LOCAL_CASPER = os.path.abspath(os.path.join(os.path.dirname(__file__), "../casper"))
if LOCAL_CASPER not in sys.path:
    sys.path.insert(0, LOCAL_CASPER)


def test_segment_init_empty():
    print("Segment defined in:", inspect.getfile(Segment))
    print("Segment.__init__ code:\n", inspect.getsource(Segment.__init__))

    segment = Segment()

    print("Type of segment.wl:", type(segment.wl))
    print("segment.wl contents:", segment.wl)

    assert isinstance(segment.wl, np.ndarray)
    assert isinstance(segment.flux, np.ndarray)
    assert segment.wl.shape == (0,)
    assert segment.flux.shape == (0,)
    assert np.isnan(segment.midpoint)
