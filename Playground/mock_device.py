

from . import shapes
import numpy as np
import logging
from Playground.array_surface_utils import Fake_CP


logging.basicConfig(format='%(asctime)s.%(msecs)03d  {%(module)s} [%(funcName)s] -- %(levelname)s: %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
log = logging.getLogger(__name__)



def build_mock_device_with_json(config):

    shapes_list = []
    for key,item in config['shapes'].items():
        shapes_list += [getattr(shapes,key)(config['ndim'], **item)]
        
    dirs = config.get('dir',None)
    origin = config.get('origin',None)
    device = Device(shapes_list,config['ndim'],origin,dirs)
        
        
    return device

# Build mock device but with fake CP approach
def build_fake_device_with_json(config):
    shapes_list = []
    for key, item in config['shapes'].items():
        shapes_list += [getattr(shapes, key)(config['ndim'], **item)]

    dirs = config.get('dir', None)
    origin = config.get('origin', None)
    device = Fake_CP_Device(shapes_list, config['ndim'], config['fake_dot_tolerances'], config['dot_locations'], origin, dirs)

    return device



class scale_for_device():
    def __init__(self,origin,dir):
        self.origin = origin
        self.dir = dir
    def __call__(self,params,bc=True):
        if bc:
            return  (params - self.origin[np.newaxis,:])*self.dir[np.newaxis,:]
        else:
            return  (params - self.origin)*self.dir

class Device():
    def __init__(self,shapes_list,ndim,origin=None,dir=None):
    
        origin = np.array([0]*ndim) if origin is None else np.array(origin)
        dir = np.array([-1]*ndim) if dir is None else np.array(dir)
        
        #Negative dir prefered so flip signs
        dir = dir*-1
        self.sd = scale_for_device(origin,dir)
        
        self.shapes_list = shapes_list
        
    def jump(self, params):
        self.params = np.array(params)[np.newaxis,:]
        return params
        
    def measure(self):
        return float(np.any([shape(self.sd(self.params)) for shape in self.shapes_list]))
        
    def check(self,idx=None):
        return self.params.squeeze() if idx is None else self.params.squeeze()[idx]
    
    def arr_measure(self,params):
        shape_logical = [shape(self.sd(params)) for shape in self.shapes_list]
        
        shape_logical = np.array(shape_logical)
        
        return np.any(shape_logical,axis=0)
        
        
class Fake_CP_Device(Device):
    def __init__(self, shapes_list, ndim, tolerance, locations, origin=None, dir=None):
        log.info("Fake_CP device initialised.")
        origin = np.array([0] * ndim) if origin is None else np.array(origin)
        dir = np.array([-1] * ndim) if dir is None else np.array(dir)


        # Negative dir preferred so flip signs
        dir = dir * -1

        print("scaling")
        self.sd = scale_for_device(origin, dir)

        self.shapes_list = shapes_list
        self.Crosstalk_shape = self.shapes_list[self.find_crosstalk_box()]
        self.fake_dots = [Fake_CP(self.Crosstalk_shape, axes=[location[0], location[1]], tolerance=tolerance[i]) for i, location in enumerate(locations)]
        self.params = list(np.zeros([ndim]))
        log.info("{} fake dots initialised".format(len(self.fake_dots)))

    def set_params(self, params):
        self.params = params
        return self.params

    def get_params(self):
        return self.params

    def jump(self, params):
        self.params = np.array(params)[np.newaxis, :]
        return self.params

    def subtune_jump(self, params):
        pass

    def subtune_measure(self):
        return float(np.any([shape(self.sd(self.params)) for shape in self.shapes_list]))


    def measure(self):
        return float(np.any([shape(self.sd(self.params)) for shape in self.shapes_list]))

    def check(self, idx=None):
        return self.params.squeeze() if idx is None else self.params.squeeze()[idx]

    def arr_measure(self, params):
        shape_logical = [shape(self.sd(params)) for shape in self.shapes_list]

        shape_logical = np.array(shape_logical)

        return np.any(shape_logical, axis=0)

    def find_crosstalk_box(self):
        # Get the index of the crosstalk box from the shapes list
        sl = self.shapes_list
        for item in sl:
            if type(item).__name__ == 'Crosstalk_matrix_box':
                return sl.index(item)

    def fake_peak_check(self, params):
        cp = np.any(np.array([dot.check_cp(params) for dot in self.fake_dots]))
        log.info('Fake CP found: {}'.format(cp))
        return cp, cp, None

    # for 2D only. Also not implemented
    def sub_peak_check(self, params):
        full_params = self.params
        full_params[:2] = params
        return self.fake_peak_check(full_params)
