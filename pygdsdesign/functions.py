from typing import Union
from pygdsdesign.polygonSet import PolygonSet
from pygdsdesign.operation import select_polygon_per_layer
import numpy as np

def print_ebeam_time(polygon: PolygonSet,
                     layer: Union[int, dict],
                     beam_current: float,
                     resist_dose: float,
                     datatype: int=0,
                     eos: int=3,
                     stage_displacement_time: float=0.05) -> None:
    """
    Print the exposure time of the layer based on its area, the beam current and
    the resist dose.
    Print the stage displacement time of the layer based on its area, the eos
    mode and the estimated stage displacement time.
    The total time is simply the sum of the exposure and displacement time.

    Args:
        polygon: Polygon from which the layer is extracted
        layer:
            if int: layer we want the ebeam exposure time from
            if dict if a key 'layer': layer we want the ebeam exposure time from
        beam_current: beam current in nA.
        resist_dose: resis dose in uC/cm2
        datatype: layer datatype.
            if argument `layer` is a dict, this argument `datatype` is ignored.
            if argument `layer` is an int, this argument `datatype` is used.
            default, 0.
        eos: EOS mode of the ebeam job.
            Must be either 3 or 6.
            An eos mode 3 imply a stage field of 500x500 um2.
            An eos mode 6 imply a stage field of 62.5x62.5 um2.
        stage_displacement_time: Time taken by the ebeam stage to move from
            field to field in second.
    """
    if isinstance(layer, dict):
        if 'layer' in layer.keys():
            if isinstance(layer['layer'], int):
                l = layer['layer']
            else:
                raise ValueError('The key "layer" of the layer argument must be an int')
        else:
            raise ValueError('Layer argument must be a dict containing a key "layer"')
        if 'datatype' in layer.keys():
            if isinstance(layer['datatype'], int):
                d = layer['datatype']
            else:
                raise ValueError('The key "datatype" of the layer argument must be an int')
        else:
            raise ValueError('Layer argument must be a dict containing a key "datatype"')
    elif isinstance(layer, int):
        l = layer
        d = datatype
    else:
        raise ValueError('Layer argument must be an int')

    # get layer area in um2
    area = select_polygon_per_layer(polygon, layer=l, datatype=d).get_area()

    # exposure time in s
    exposure_time = area/1e8*resist_dose*1e3/beam_current
    minutes = exposure_time//60
    hours = minutes//60

    print('')
    print('++++++++++++++++++++++++++++++++++++')
    print('ebeam info:')
    print('    resist sensitivity: {} uC/cm2'.format(resist_dose))
    print('    area: {:.0f} um2 = {:.1f} mm2'.format(area, area/1e6))
    print('    current: {} nA'.format(beam_current))
    print('    exposure duration: {:2.0f}h {:2.0f}min {:2.0f}s'.format(hours, minutes % 60, exposure_time % 60))

    # get layer containing area in um2
    dx, dy = polygon.get_size()
    area_c = dx*dy
    # stage displacement time
    if eos==3:
        field_area = 500.*500.
    elif eos==6:
        field_area = 62.5*62.5
    else:
        raise ValueError('eos argument must be either 3 or 6')

    stage_time = area_c/field_area*stage_displacement_time
    minutes = stage_time//60
    hours = minutes//60
    print('    number of field: {:.0f}'.format(area_c/field_area))
    print('    stage displacement duration: {:2.0f}h {:2.0f}min {:2.0f}s'.format(hours, minutes % 60, stage_time % 60))

    # total time
    total_time = exposure_time + stage_time
    minutes = total_time//60
    hours = minutes//60
    print('    total ebeam duration: {:2.0f}h {:2.0f}min {:2.0f}s'.format(hours, minutes % 60, total_time % 60))




def distance(x1:float,y1:float,x2:float,y2:float) -> float:
            """Return the distance between the two point (x1;y1) and (x2;y2)"""
            return np.sqrt((x1-x2)**2 + (y2-y1)**2)