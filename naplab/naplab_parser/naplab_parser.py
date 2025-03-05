import os
import numpy as np
import copy

from naplab.naplab_parser.parsers.gnss_parser import GNSSParser
from naplab.naplab_parser.parsers.cam_parser import CamParser
import errno

from naplab.naplab_parser.utils import parsing_utils, timestamps_utils, transformation_utils, f_theta_utils
from naplab.naplab_parser.tables import Scene, Sample, SampleData, CalibratedSensor, EgoPose, NabLabMap



class NapLabParser: 

    def __init__(self, raw_dataroot, trip, processed_dataroot, selected_cams, nuscnes_path=False, nbr_samples=40):

        if not os.path.exists(raw_dataroot := os.path.join(raw_dataroot, trip)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), raw_dataroot)
        
        if not os.path.exists(processed_dataroot):
            os.makedirs(processed_dataroot)

        self.trip = trip
        self.raw_dataroot = raw_dataroot
        self.processed_dataroot = os.path.join(processed_dataroot, self.trip) 
        self.selected_cams = selected_cams
        self.start_cam_index = None
        self.end_gnss_index = None
        self.nuscnes_path = nuscnes_path

        self.set_nbr_samples_in_scenes(nbr_samples=nbr_samples)
        
       
        self.load_paths()
        cams_timestamps = parsing_utils.get_cams_timestamps(self.cams_timestamps_files)

        
        self.lat_lon, self.gnss_timestamps = GNSSParser.get_gnns_data(self.gnss_file[0])

        self.prep_cams_timestamps, self.gnss_timestamps = timestamps_utils.preparare_timestsamps(cams_timestamps, self.gnss_timestamps[:]) # all same length
        # Trip077 len cam 16 570, len gnss 5524 * 3 = 16 572  # len(self.prep_cams_timestamps['C6_L1'])

        self.get_average_sync_diff(self.prep_cams_timestamps, self.gnss_timestamps)
        self.res = timestamps_utils.get_best_syncs(self.prep_cams_timestamps, self.gnss_timestamps, output_path_plot=self.processed_dataroot, save=True)

        self.set_start_stop_indexes('C4_rearCam')

        
        self.naplab2nuscenes = {
        'C1_front60Single': 'CAM_FRONT',
        'C8_R2': 'CAM_FRONT_RIGHT', 
        'C7_L2': 'CAM_FRONT_LEFT', 
        'C4_rearCam': 'CAM_BACK', 
        'C6_L1': 'CAM_BACK_LEFT', 
        'C5_R1':  'CAM_BACK_RIGHT'
        }

        if self.nuscnes_path:
            CamParser.load_nuscenes_camera_parameters(self.nuscnes_path)
            CamParser.set_naplab2nuscens(self.naplab2nuscenes)


        self.camera_parameters = CamParser.get_cams_parameters(self.camera_parameters_file[0], self.selected_cams)
        self.ego_xy_position = GNSSParser.get_ego_xy_positions(self.lat_lon)

        self.bearings = GNSSParser.compute_bearing(self.lat_lon)

    def set_nbr_samples_in_scenes(self, nbr_samples):
        self.nbr_samples = nbr_samples 
        GNSSParser.set_nbr_samples(nbr_samples=nbr_samples)
        

    def load_paths(self):

        self.absoulte_files = parsing_utils.get_subfolders(self.raw_dataroot)
        self.camera_parameters_file = parsing_utils.get_files(self.absoulte_files, file_format="json", file_key="camera")
        self.gnss_file = parsing_utils.get_files(self.absoulte_files, file_format="bin", file_key="gnss50")
        self.cams_timestamps_files = parsing_utils.get_files(self.absoulte_files, file_format="timestamps", selected_cams=self.selected_cams)
        self.cams_files = parsing_utils.get_files(self.absoulte_files, file_format="h264", selected_cams=self.selected_cams)
    
    def set_start_stop_indexes(self, cam_name): 

        print("Updating start cams and end gnss timestamps...")
        self.start_cam_index = self.res[cam_name]['arg1_index_start']
        self.end_gnss_index = self.res[cam_name]['arg2_index_end']

        self.adjust_timestamps()

    def get_average_sync_diff(self, cams_timestamps, gnss_timestamps): 

        results = []
        for cam_name, cam_timestamps in cams_timestamps.items():
            freq_ratio = round(timestamps_utils.get_freq_ration(cam_timestamps,gnss_timestamps))
            sec = timestamps_utils.get_sync_diff(cam_timestamps[::freq_ratio], gnss_timestamps)
            results.append(sec)

        average_sec_diff = np.mean(np.abs(np.array(results)), axis=-1)

        self.freq_ratio = freq_ratio # for use later, assue all cams are the same now

        print(f"----------- Average Synce Diff To GNSS: {average_sec_diff} seconds | Cam Start Index {self.start_cam_index} | GNSS End Index {self.end_gnss_index} | ------------------")

    def adjust_timestamps(self): 
        
        print("Adjusting timestamps to new starts and end...")
        best_cams_timestamps = {}
        for cam_name, cam_timestamps in self.prep_cams_timestamps.items():

            best_cams_timestamps[cam_name] = (cam_timestamps[self.start_cam_index:])
        
        best_gnss_timestamps = self.gnss_timestamps[:self.end_gnss_index]

        self.best_cams_timestamps = best_cams_timestamps
        self.best_gnss_timestamps = best_gnss_timestamps

        self.get_average_sync_diff(self.best_cams_timestamps, self.best_gnss_timestamps)


    def visualize_route(self, save=False, scenes=False):

        GNSSParser.plot_route(self.lat_lon[:], processed_dataroot=self.processed_dataroot, title=self.trip, save=save, freq_ratio=self.freq_ratio, scenes=scenes)

    def visualize_route_bearings(self, refrence_frame=False, scenes=False, save=False):

        GNSSParser.plot_route_bearings(bearings=self.bearings[:], ego_xy_positions=self.ego_xy_position[:], freq_ratio=self.freq_ratio, refrence_frame=refrence_frame, scenes=scenes, save=save, processed_dataroot=self.processed_dataroot)
    
    def visualize_sync_diffs(self, save=False):

        timestamps_utils.plot_sync_diff(self.res, save=save)

    def extract_images(self, scenes=False):

        cams_samples_path = os.path.join(self.processed_dataroot, 'samples')
        cam_names = list(self.camera_parameters.keys())

        cam_folders = parsing_utils.create_cam_folders(cams_samples_path, cam_names)

        if scenes:
            samples_idx = GNSSParser.samples_idx_from_scenes(scenes=scenes, max_len=len(self.best_gnss_timestamps))
            frames_idx = samples_idx * self.freq_ratio # one sample every third frame
            self.extract_images_samples_idx(frames_idx, cam_folders, self.cams_files[:])

        else: 
            print("")


    def extract_images_samples_idx(self, frames_idx, cam_folders, cams_files): 

        
        for cam_file, cam_folder, (cam_name, cam_timestamps) in zip(cams_files, cam_folders, self.best_cams_timestamps.items()): 
            CamParser.save_images_from_camera(cam_file, cam_folder, cam_name, cam_timestamps, frames_idx, self.freq_ratio)


    def load_cams_timestamps(self):
        pass

    def create_database(self, description):
        
        naplab_map = self.create_naplab_map(description=description)
        cb_sensors = self.create_calibrated_sensor()
        ego_poses = self.create_ego_pose()
        scenes, samples, samples_data = self.create_samples(naplab_map=naplab_map, calibrated_sensors=cb_sensors, ego_poses=ego_poses)

        parsing_utils.save_tables_json(self.processed_dataroot, naplab_map, "naplab_map.json")
        parsing_utils.save_tables_json(self.processed_dataroot, cb_sensors, "calibrated_sensors.json")
        parsing_utils.save_tables_json(self.processed_dataroot, ego_poses, "ego_poses.json")
        parsing_utils.save_tables_json(self.processed_dataroot, scenes, "scenes.json")
        parsing_utils.save_tables_json(self.processed_dataroot, samples, "samples.json")
        parsing_utils.save_tables_json(self.processed_dataroot, samples_data, "samples_data.json")
        print("Finished saving tables!")


    def create_naplab_map(self, description): 

        naplab_maps = []
        naplab_map = NabLabMap(trip=self.trip, description=description, lat_lon_coordinates=self.lat_lon, bearings=list(self.bearings))
        naplab_maps.append(naplab_map.__dict__)
        
        return naplab_maps


    def create_calibrated_sensor(self):
        cd_sensors = []

        for i, (k, v) in enumerate(self.camera_parameters.items()):
            cd_sensor = CalibratedSensor(translation=v['t'], rotation=transformation_utils.euler_to_quaternion_yaw(v['roll_pitch_yaw']),\
                 cx=v['cx'], cy=v['cy'], fw_coeff=v['fw_coeff'], fw_coeff_0_start=v['fw_coeff_0_start'], bw_coeff=v['bw_coeff'], nuscene_camera_intrinsics=v['nuscenes_cam_intrinsics'], nuscene_image_size=v['nuscenes_image_size'])
            cd_sensors.append(cd_sensor.__dict__)

        return cd_sensors


    def create_ego_pose(self, refrence_frame_bearings=500): 
        ego_poses = []

        x_translation = self.ego_xy_position.x - self.ego_xy_position.x[0]
        y_translation = self.ego_xy_position.y - self.ego_xy_position.y[0]


        dir_bearings = self.bearings - self.bearings[refrence_frame_bearings]

        for i, gnss_timestamp in enumerate(self.best_gnss_timestamps): 
            ego_pose = EgoPose(translation=[float(x_translation[i]), float(y_translation[i]), float(0)], rotation=transformation_utils.euler_to_quaternion_yaw([0,0,dir_bearings[i]]), timestamp=gnss_timestamp)
            ego_poses.append(ego_pose.__dict__)
        

        return ego_poses
    

    def prepare_cam_timestaps_freq(self, cams_timestamps): 

        new_dict = {}

        for k, v in cams_timestamps.items(): 
            new_dict[k] = v[::3]

        return new_dict

    def generate_filenmae(self, cam, timestamp):
        return os.path.join(self.processed_dataroot, "samples", cam, f"{cam}_{timestamp}.png")
    
    def create_samples(self, naplab_map, calibrated_sensors, ego_poses):

        samples = []
        scenes = []
        samples_data = []
        scene_count = 0
        freq_best_cams_timestamps = self.prepare_cam_timestaps_freq(self.best_cams_timestamps)

        for i, timestamp_gnss in enumerate(self.best_gnss_timestamps): 
            
            sample = Sample(timestamp=timestamp_gnss)

            if i % self.nbr_samples == 0:
                scene = Scene(naplab_map_token=naplab_map[0]['token'], scene_name=f"scene_{scene_count}", nbr_samples=(40 if i%40 == 0 else i%40), dataroot=self.raw_dataroot)
                scenes.append(scene.__dict__)
                scene.first_sample_token = sample.token
                scene_count += 1
            
            sample.scene_token = scene.token
            sample.scene_name =scene.scene_name
           
            data = {}


            for j, (cam_name, timestamps_cam) in enumerate(freq_best_cams_timestamps.items()):

                sample_data = SampleData(sample_token=sample.token, calibrated_sensor_token=calibrated_sensors[j]['token'],  ego_pose_token=ego_poses[i]['token'], timestamp=int(timestamps_cam[i]), filename=self.generate_filenmae(cam_name, timestamps_cam[i]), next_idx=None, prev_idx=None)
                samples_data.append(sample_data.__dict__)

                data[cam_name] = sample_data.token
            
            sample.data = data
            samples.append(sample.__dict__)

        return scenes, samples, samples_data

        
if __name__ == "__main__": 

    raw_dataroot = "/cluster/home/terjenf/MapTR/NAP_raw_data"
    trip = "Trip083"
    processed_dataroot = "/cluster/home/terjenf/naplab/data"
    nuscene_path = "/cluster/home/terjenf/naplab/naplab/naplab_parser/metadata_nuscenes/nuscene_metadata.json"

    selected_cams = [
        'C1_front60Single', 
        'C8_R2', 
        'C7_L2',
        'C4_rearCam', 
        'C6_L1',
        'C5_R1']

    naplab_parser = NapLabParser(raw_dataroot=raw_dataroot, trip=trip, processed_dataroot=processed_dataroot, selected_cams=selected_cams, nuscnes_path=nuscene_path)

    # naplab_parser.visualize_route(save=True)
    # scenes_plot = [50,51,52,53,54,55,56,57,58,59,60, 105,106,107,108,109,110]
    # naplab_parser.visualize_route(save=True, scenes=scenes_plot)
    # #naplab_parser.visualize_route(save=True, scenes=(50,60))

    # # naplab_parser.visualize_route_bearings(refrence_frame=500, save=True)
    # naplab_parser.visualize_route_bearings(refrence_frame=500, save=True)
    # naplab_parser.visualize_route_bearings(refrence_frame=0, save=True)


    # Front
    bw_coeff = naplab_parser.camera_parameters['C1_front60Single']['bw_coeff']
    w = naplab_parser.camera_parameters['C1_front60Single']['width']
    
    fw_coeff_list, thetas, r = f_theta_utils.get_fw_coeff_start_0(bw_coeff=bw_coeff, w=w, return_all=True)
    f_theta_utils.plot_forward_only(fw_coeff_list, thetas, r, output_path=naplab_parser.processed_dataroot, cam_name="C1_front60Single", save=True)



    fw_coeff_list, thetas, r = f_theta_utils.get_fw_coeff_start_sigle_0(bw_coeff=bw_coeff, w=w, return_all=True)
    f_theta_utils.plot_forward_only(fw_coeff_list, thetas, r, output_path=naplab_parser.processed_dataroot, cam_name="C1_front60Single", save=True)


    naplab_parser.visualize_route(save=True)

    # bw_coeff = naplab_parser.camera_parameters['C7_L2']['bw_coeff']
    # w = naplab_parser.camera_parameters['C1_front60Single']['width']
    

    # fw_coeff_list, thetas, r = f_theta_utils.get_fw_coeff_start_0(bw_coeff=bw_coeff, w=w, return_all=True)

    # f_theta_utils.plot_forward(fw_coeff_list, thetas, r, output_path=naplab_parser.processed_dataroot, cam_name="C7_L2", save=True)



    # f_theta_utils.plot_forward(fw_coeff_list, thetas, r, output_path=naplab_parser.processed_dataroot, cam_name="C7_L2", save=True)
    # naplab_parser.extract_images(scenes=(105,110))

    #description="Handels -> Eglseterbru -> Nidarosdomen -> Samfundet -> Høyskoleringen"
    description="Handels"

    naplab_parser.create_database(description=description)
