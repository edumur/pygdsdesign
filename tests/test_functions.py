from pygdsdesign import *
import json
import pytest




with open('polygons.json') as f:
    polygons = json.load(f)


with open('results.json') as f:
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
def test_translate(square,cross,triangle):

    #Check a basic translate
    result = square.translate(10,10)
    assert np.isclose(np.asarray([[[0,0],[0,20],[20,20],[20,0]]]), np.asarray(result.polygons)).all()

    #Check a translate of 0
    result = square.translate(0,0)
    assert np.isclose(np.asarray(square.polygons), np.asarray(result.polygons)).all()

    #Check huge translate
    result = cross.translate(9999999999,9999999999)
    assert np.isclose(np.asarray([[[9.99999998e+09,1.00000000e+10],[9.99999998e+09,1.00000000e+10],[1.00000000e+10,1.00000000e+10],[1.00000000e+10,1.00000000e+10]],[[1.00000000e+10,9.99999998e+09],[1.00000000e+10,1.00000000e+10],[1.00000000e+10,1.00000000e+10],[1.00000000e+10,9.99999998e+09]]]), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_crop(square,cross,triangle):

    #Check a basic crop
    result = crop(cross,'top',10)
    assert np.isclose(np.asarray([[[2.5,-2.5],[17.5,-2.5],[17.5,2.5],[2.5,2.5],[2.5,7.5],[-2.5,7.5],[-2.5,2.5],[-17.5,2.5],[-17.5,-2.5],[-2.5,-2.5],[-2.5,-17.5],[2.5,-17.5]]]), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_flip(square,cross,triangle):

    #Check x flip
    cross_copy = PolygonSet(cross.polygons)
    result = cross_copy.flip('x')
    assert np.isclose(np.asarray([[[-17.5,2.5],[-17.5,-2.5],[17.5,-2.5],[17.5,2.5]],[[-2.5,17.5],[-2.5,-17.5],[2.5,-17.5],[2.5,17.5]]]), np.asarray(result.polygons)).all()

    #Check y flip
    cross_copy = PolygonSet(cross.polygons)
    result = cross_copy.flip('y')
    assert np.isclose(np.asarray([[[17.5,-2.5],[17.5,2.5],[-17.5,2.5],[-17.5,-2.5]],[[2.5,-17.5],[2.5,17.5],[-2.5,17.5],[-2.5,-17.5]]]), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_center(square,cross,triangle):

    #Check x flip
    poly = PolygonSet([[(1,2),(3,4)]])
    result = poly.center()
    assert np.isclose(np.asarray([[[-1., -1.],[ 1.,  1.]]]), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_get_center(square,cross,triangle):

    #Check x flip
    poly = PolygonSet([[(1,2),(3,4)]])
    result = poly.get_center()
    assert result == (2.0, 3.0) 



@pytest.mark.usefixtures("square", "cross", "triangle")
def test_get_bounding_box(square,cross,triangle):

    result = triangle.get_bounding_box()
    assert np.isclose(np.asarray([[-10, -10],[ 10,  10]]), np.asarray(result)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_get_area(square,cross,triangle):

    result = cross.get_area()
    assert 350.0 == result

@pytest.mark.usefixtures("square", "cross", "triangle")
def test_rotate(square,cross,triangle):

    result = cross.rotate(99)
    assert np.isclose(np.asarray([[[-3.19488249,17.3865674],[1.80115168,17.5856718],[3.19488249,-17.3865674],[-1.80115168,-17.5856718]],[[-17.5856718,1.80115168],[17.3865674,3.19488249],[17.5856718,-1.80115168],[-17.3865674,-3.19488249]]]), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_scale(square,cross,triangle):

    result = cross.scale(99,99)
    assert np.isclose(np.asarray([[[-1732.5,-247.5],[-1732.5,247.5],[1732.5,247.5],[1732.5,-247.5]],[[-247.5,-1732.5],[-247.5,1732.5],[247.5,1732.5],[247.5,-1732.5]]]), np.asarray(result.polygons)).all()

@pytest.mark.usefixtures("square", "cross", "triangle")
def test_scale(square,cross,triangle):

    result = cross.mirror((-10,10), (10,-10))
    assert np.isclose(np.asarray([[[2.5,17.5],[-2.5,17.5],[-2.5,-17.5],[2.5,-17.5]],[[17.5,2.5],[-17.5,2.5],[-17.5,-2.5],[17.5,-2.5]]]), np.asarray(result.polygons)).all()


@pytest.mark.usefixtures("square", "cross", "triangle")
def test_change_layer(square,cross,triangle):

    #Check a basic change_layer
    poly = PolygonSet(triangle.polygons, [1],[2],["test"],["blue"])
    result = poly.change_layer(2,3,"sucess1","sucess2")
    assert np.isclose(np.asarray([[[-10,-10],[0,10],[10,-10]]]), np.asarray(result.polygons)).all()
    assert result.layers==[2]
    assert result.datatypes==[3]
    assert result.colors==["sucess1"]
    assert result.names==["sucess2"]

    #Check a change_layer on a empty polygon
    poly = PolygonSet()
    result = poly.change_layer(2,3,"sucess1","sucess2")
    assert np.isclose(np.asarray([[0,0]]), np.asarray(result.polygons)).all()
    assert result.layers==[2]
    assert result.datatypes==[3]
    assert result.colors==["sucess1"]
    assert result.names==["sucess2"]


# TODO : correct this test
# @pytest.mark.usefixtures("square", "cross", "triangle")
# def test_remove_polygon_from_layer(square,cross,triangle):

#     poly1 = PolygonSet(triangle.polygons, [1],[2],["test"],["blue"])
#     poly2 = PolygonSet(triangle.polygons, [2],[2],["test"],["blue"])
#     poly3 = PolygonSet(triangle.polygons, [2],[2],["test"],["blue"])
#     poly1 += poly2
#     poly1 += poly3
#     result = poly1.remove_polygon_from_layer(2)
#     assert np.isclose(np.asarray([[[-10,-10],[0,10],[10,-10]]]), np.asarray(result.polygons[0])).all()
