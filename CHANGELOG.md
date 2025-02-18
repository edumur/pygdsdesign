pygdsdesign-0.17.5

  Bug fixes:
    - bugfix of layer 42 of the qubit mask. Removed crosses that were not existing in reality

pygdsdesign-0.17.4

  Bug fixes:
    - Fix `print_ebeam_time` when the requested datatype is not "0"
    - Fix `grid_cover` when the only_square was used with only one square

pygdsdesign-0.17.3

  New Features:
    - Add `hexagonal_grid` input parameter to `grid_cover` operation

  Bug fixes:
    - Fix `add_turn` total_length computation
    - Fix `rotate` for transmission line
      self.ref takes now the rotation into account
    - Propagate `self.ref` for MicrostripPolar
    - Fix `translate` for transmission line
      if `_bounding_polygon` exist, propagate the translation to it
    - Fix `export_gds` when the same layer may have different datatype

pygdsdesign-0.17.2

  New Features:
    - Add `RectangleCentered` polygon and `qubit_layer_19` shape

pygdsdesign-0.17.1

  Bug Fixes:
    - Fix "layer" issue in PolygonSet
    - Propagate `ref` to Microstrip and CPW
    - Fix `shapes` file with new functions name

  New Features:
    - Add `CPWPolar` and `MicrostripPolar` class which allow easy diagonal lines
    - Add `select_polygon_per_name` operation
    - Add `noise` and `only_square` parameters to `grid_cover` operation
    - Add `fillet` method
