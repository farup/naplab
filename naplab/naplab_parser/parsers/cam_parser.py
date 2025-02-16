
import json
from naplab.naplab_parser.utils import f_theta_utils

class CamParser: 

    @classmethod
    def set_naplab2nuscens(cls, naplab2nuscene):
        cls.naplab2nuscenes = naplab2nuscenes 

    @classmethod
    def load_nuscenes_camera_parameters(cls, nuscenes_cam_path): 
        with open(nuscenes_cam_path, 'r') as file:
            cls.nuscenes_cam = json.load(f)

    @staticmethod
    def extract_images(trip, scenes=False): 
        pass
    
    def get_nuscenes_cam_intrinsics(naplab_cam):
        cam_intrinsics = CamParser.naplab2nuscenes[naplab_cam]['camera_intrinsic']
        image_size = CamParser.naplab2nuscenes[naplab_cam]['img_size']
        return cam_intrinsics, image_size

    @staticmethod
    def get_cams_parameters(calibrated_sesnsor_file, selected_cams=False): 
        """ Return """

        with open(calibrated_sesnsor_file, 'r') as file:
            json_obj = json.load(file)

            cam_data = {}
           
            for car_mask in json_obj['rig']['sensors']: 
                try: 
                    if 'car-mask' not in car_mask.keys():
                        continue
                    
                    if selected_cams:
                        if car_mask['name'] not in selected_cams:
                            continue

                    nominalSensor2Rig_FLU = car_mask['nominalSensor2Rig_FLU']
                
                    roll_pitch_yaw = nominalSensor2Rig_FLU['roll-pitch-yaw']
                    t = nominalSensor2Rig_FLU['t']
                    properties = car_mask['properties']

                    cx = float(properties['cx'])
                    cy = float(properties['cy'])

                    height = int(properties['height'])
                    width = int(properties['width'])

                    bw_poly = (properties['bw-poly']).split(" ")
                    bw_coeff = [float(num) for num in bw_poly if len(num) > 2]

                    fw_coeff = f_theta_utils.get_fw_coeff(width, bw_coeff)

                    nuscenes_cam_intrinsics, nusecnes_image_size = get_nuscenes_cam_intrinsics(car_mask['name'])

                    cam_data[car_mask['name']] = {'roll_pitch_yaw': roll_pitch_yaw, 't':t, 'cx': cx, 'cy': cy, \
                        'height': height, 'width': width, 'bw_coeff': bw_coeff, \
                            'fw_coeff': fw_coeff, 'nuscenes_cam_intrinsics': nuscenes_cam_intrinsics, 'nusecnes_image_size': nusecnes_image_size}

                except KeyError as e:
                    print("Error", e)
                    print(car_mask)

        return cam_data