import json 
import uuid
from dataclasses import dataclass, field

import pickle



def generate_token():
    return str(uuid.uuid4())


@dataclass
class NabLabMap: 
    trip: str
    description: str
    lat_lon_coordinates: list
    bearings: list
    token: str = field(init=False)  # Excluded from __init__, set later

    def __post_init__(self):
        self.token = generate_token()  

@dataclass
class Scene: 
    scene_name: str
    nbr_samples: int
    dataroot: str
    naplab_map_token: str
    first_sample_token: str = field(default="")
    last_sample_token: str = field(default="")
    map_token: str = field(default="")
    token: str = field(init=False)  # Excluded from __init__, set later

    def __post_init__(self):
        self.token = generate_token()  # Generate token after initialization

@dataclass      
class Sample: 
    timestamp: int
    scene_token: any = field(default=None)
    scene_name: any = field(default=None)
    data: any = field(default=None) # default to None, can be set later
    next_idx: any = field(default=None)
    prev_idx: any = field(default=None)
    
    def __post_init__(self):
        self.token = generate_token()  # Generate token after initialization

@dataclass 
class SampleData: 
    sample_token: int
    ego_pose_token: int
    timestamp: int
    calibrated_sensor_token: int
    filename: str
    next_idx: str
    prev_idx: str

    def __post_init__(self):
        self.token = generate_token()  # Generate token after initialization


@dataclass
class CalibratedSensor:
    """
    [copy from nuscenes]
    Definition of a particular sensor (lidar/radar/camera) as calibrated on a particular vehicle. 
    All extrinsic parameters are given with respect to the ego vehicle body frame.
    All camera images come undistorted and rectified.
    
    """
    translation: list
    rotation: list
    nuscene_camera_intrinsics: list
    nuscene_image_size: list
    fw_coeff: list 
    fw_coeff_0_start: list 
    bw_coeff: list
    cx: int
    cy: int 

    def __post_init__(self):
        self.token = generate_token()  # Generate token after initialization

@dataclass
class EgoPose: 

    """"
    Ego vehicle pose at a particular timestamp. Given with respect to global coordinate system of the log's map. 
    The localization is 2-dimensional in the x-y plane.
    """
    translation: list 
    rotation: list
    timestamp: int
    lat: any= field(default=None)
    lon: any = field(default=None)

    def __post_init__(self):
        self.token = generate_token()  # Generate token after initialization


  
# class Log: 
#     def __init__(self, vehicle_name, date_caputred, location):
#         self.token = generate_token()
#         self.vehicle_name = vechile_name
#         self.date_captured = date_caputred
#         self.location = location


# class Sensor: 
#     def __init__(self, channel, modility):
#         self.token = generate_token()
#         self.channel = channel
#         self.modality = channel