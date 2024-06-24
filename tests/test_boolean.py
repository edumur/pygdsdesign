from pygdsdesign import *
import json
import os
import pytest

PATH = os.path.dirname(os.path.realpath(__file__))


with open(os.path.join(PATH, 'polygons.json')) as f:
    polygons = json.load(f)


with open(os.path.join(PATH, 'results.json')) as f:
    results = json.load(f)


@pytest.fixture
def square():
    return PolygonSet(polygons['square']['points'])


@pytest.fixture
def cross():
    return PolygonSet(polygons['cross']['points'])


@pytest.fixture
def triangle():
    return PolygonSet(polygons['triangle']['points'])


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_boolean(square,cross,triangle):

    # Check basic NOT operation
    result = boolean(square,cross, "not")
    assert np.isclose(np.asarray(results['boolean']['1']), np.asarray(result.polygons)).all()

    # Check basic NOT followed by OR
    temp = boolean(square,cross, "not")
    result = boolean(temp,triangle, "or")
    assert np.isclose(np.asarray(results['boolean']['2']), np.asarray(result.polygons)).all()

    # Check NOT with the same polygon
    result = boolean(cross,cross, "not")
    assert result == None

    # Check AND with the same polygon
    result = boolean(cross,cross, "and")
    assert np.isclose(np.asarray(result.polygons), np.asarray(result.polygons)).all()

    # Check OR with the same polygon
    result = boolean(cross,cross, "or")
    assert np.isclose(np.asarray(result.polygons), np.asarray(result.polygons)).all()

    # Check a basic XOR
    result = boolean(cross, triangle, "xor")
    for i in range(len(results['boolean']['3'])):
        assert np.isclose(np.asarray(results['boolean']['3'][i]), np.asarray(result.polygons[i])).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_offset(square,cross,triangle):

    # Check a basic offset
    result = offset(cross, 10)
    assert np.isclose(np.asarray(results['offset']['1']), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_inverse_polarity(square,cross,triangle):

    # Check a basic inverse_polarity
    result = inverse_polarity(cross)
    for i in range(len(results['inverse_polarity']['1'])):
        assert np.isclose(np.asarray(results['inverse_polarity']['1'][i]), np.asarray(result.polygons[i])).all()
