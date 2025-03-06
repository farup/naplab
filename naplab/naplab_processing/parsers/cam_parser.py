
import json
import cv2
import os


from naplab_processing.utils import f_theta_utils

class CamParser: 
   

    nuscenes_cam = False
    nuscenes_cam_intrinsics = None
    nusecnes_image_size = None

    @classmethod
    def set_naplab2nuscens(cls, naplab2nuscens):
        cls.naplab2nuscenes = naplab2nuscens

    @classmethod
    def load_nuscenes_camera_parameters(cls, nuscenes_cam_path): 
        with open(nuscenes_cam_path, 'r') as file:
            cls.nuscenes_cam = json.load(file)



    @staticmethod
    def save_images_from_camera(cam_file, cam_folder, cam_name, cam_timestamps, frames_idx, freq_ratio):

        """
        Extraxt and save images.

        Aargs:
            cam_folder_path: (str) folder to save extracted images
            camera_file: (str) path to camera file
            time_stamps_cam: corresponding timestamps for the camera_file
    
        """

        cam_cap = cv2.VideoCapture(cam_file)
        print("FPS: ", cam_cap.get(cv2.CAP_PROP_FPS))


        if not cam_cap.isOpened():
            print("Error: Unable to open the .h264 file")
        else:
            frame_count = 0
            while True:

                ret, frame = cam_cap.read()
                if not ret:
                    break
                
                if frame_count not in frames_idx:
                    frame_count += 1
                    continue

                if frame_count > max(frames_idx):
                    break

                # Save the frame as an image file

                if ((frame_count - min(frames_idx)) % freq_ratio) != 0: 
                    frame_count += 1
                    continue

                frame_filename = os.path.join(cam_folder, f"{cam_name}_{cam_timestamps[frame_count]}.png")
                frame_count += 1

                if os.path.exists(frame_filename): 
                    continue

                cv2.imwrite(frame_filename, frame)
                print(f"Saved: {frame_filename}")

                
                # if frame_count >= 550:
                #     break
            # Release resources
            cam_cap.release()

            print(f"Extracted {frame_count} frames to {cam_folder}")



    
    def get_nuscenes_cam_intrinsics(naplab_cam):
        cam_intrinsics =  CamParser.nuscenes_cam[CamParser.naplab2nuscenes[naplab_cam]]['camera_intrinsic']
        image_size =  CamParser.nuscenes_cam[CamParser.naplab2nuscenes[naplab_cam]]['img_size']

        return cam_intrinsics, image_size

    @staticmethod
    def get_cams_parameters(calibrated_sesnsor_file, selected_cams=False): 
        """ Return """

        nuscenes_cam_intrinsics = None
        nusecnes_image_size = None

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
                    fw_coeff_0_start = f_theta_utils.get_fw_coeff_start_0(width, bw_coeff)

                    if CamParser.nuscenes_cam:
                        nuscenes_cam_intrinsics, nusecnes_image_size = CamParser.get_nuscenes_cam_intrinsics(car_mask['name'])

                    cam_data[car_mask['name']] = {'roll_pitch_yaw': roll_pitch_yaw, 't':t, 'cx': cx, 'cy': cy, \
                        'height': height, 'width': width, 'bw_coeff': bw_coeff, \
                            'fw_coeff': fw_coeff, 'fw_coeff_0_start': fw_coeff_0_start, 'nuscenes_cam_intrinsics': nuscenes_cam_intrinsics, 'nuscenes_image_size': nusecnes_image_size}

                except KeyError as e:
                    print("Error", e)
                    print(car_mask)

        return cam_data