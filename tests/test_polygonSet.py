from pygdsdesign import PolygonSet, GdsLibrary, Text, MicroStrip, Rectangle
import numpy as np


def test_constructor():
    # Check if we can initialize an empty polygon correctly
    result = PolygonSet()
    assert np.isclose(np.asarray(result.polygons), np.asarray([[[0, 0]]])).all()
    assert result.layers == [0]
    assert result.datatypes == [0]
    assert result.names == ['']
    assert result.colors == ['']

    # Check the Auto-completion of the arguments based on polygons
    result = PolygonSet([[(1,2),(3,4)],[(1,2),(3,4)]])
    assert np.isclose(np.asarray(result.polygons), np.asarray([[[1,2],[3,4]]])).all()
    assert result.layers == [0,0]
    assert result.datatypes == [0,0]
    assert result.names == ['','']
    assert result.colors == ['','']

    # Check the Auto-completion of the arguments based on layers
    result = PolygonSet([[(1,2),(3,4)]], [0,0,0])
    assert np.isclose(np.asarray(result.polygons), np.asarray([[[1,2],[3,4]]])).all()
    assert result.layers == [0,0,0]
    assert result.datatypes == [0,0,0]
    assert result.names == ['','', '']
    assert result.colors == ['','', '']


def test_polygons_addition():

    # Check a basic polygon addition
    result = PolygonSet([[(1,2),(3,4)]])
    temp = PolygonSet([[(1,2),(3,4)],[(1,2),(3,4)]])
    result = temp + result
    oracle = [[[1,2],[3,4]],[[1,2],[3,4]],[[(1,2),(3,4)]]]

    for i in range(len(oracle)):
        assert np.isclose(np.asarray(oracle[i]), np.asarray(result.polygons[i])).all()
    assert result.layers == [0,0,0]
    assert result.datatypes == [0,0,0]
    assert result.names == ['','','']
    assert result.colors == ['','','']
