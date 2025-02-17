import os
import numpy as np

from naplab.naplab_parser.parsers.gnss_parser import GNSSParser
from naplab.naplab_parser.parsers.cam_parser import CamParser
import errno

from naplab.naplab_parser.utils import parsing_utils, timestamps_utils, transformation_utils
from naplab.naplab_parser.tables import Scene, Sample, SampleData, CalibratedSensor, EgoPose, Map



class NapLabParser: 

    def __init__(self, raw_dataroot, trip, processed_dataroot, selected_cams, nuscnes_path=False, nbr_samples=40):

        if not os.path.exists(raw_dataroot := os.path.join(raw_dataroot, trip)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), raw_dataroot)
        
        if not os.path.exists(processed_dataroot):
            os.makedirs(processed_dataroot)

        self.trip = trip
        self.raw_dataroot = raw_dataroot
        self.processed_dataroot = processed_dataroot
        self.selected_cams = selected_cams
        self.start_cam_index = None
        self.end_gnss_index = None
        self.nuscnes_path = nuscnes_path

        GNSSParser.set_nbr_samples(nbr_samples=nbr_samples)
       
        self.load_paths()
        cams_timestamps = parsing_utils.get_cams_timestamps(self.cams_timestamps_files)

        self.prep_cams_timestamps = timestamps_utils.preparare_timestsamps(cams_timestamps) # all same length
        self.lat_lon, self.gnss_timestamps = GNSSParser.get_gnns_data(self.gnss_file[0])


        self.get_average_sync_diff(self.prep_cams_timestamps, self.gnss_timestamps)
        self.res = timestamps_utils.get_best_syncs(self.prep_cams_timestamps, self.gnss_timestamps)

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
            freq_ratio = round(timestamps_utils.get_freq_ration(cam_timestamps,gnss_timestamps ))
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


    def visualize_route(self, processed_dataroot):

        GNSSParser.plot_route(self.lat_lon[:], processed_dataroot=processed_dataroot)

    def visualize_route_bearings(self, refrence_frame=False, processed_dataroot=False, scenes=False): 

        GNSSParser.plot_route_bearings(bearings=self.bearings[:], ego_xy_positions=self.ego_xy_position[:], freq_ratio=self.freq_ratio, refrence_frame=refrence_frame, processed_dataroot=processed_dataroot, scenes=scenes)
    
    def visualize_sync_diffs(self, save=False):

        timestamps_utils.plot_sync_diff(self.res, save=save)


    def extract_images(self, scenes=False):

        cams_samples_path = os.path.join(self.processed_dataroot, self.trip, 'samples')
        cam_names = list(self.camera_parameters.keys())

        cam_folders = parsing_utils.create_cam_folders(cams_samples_path, cam_names)

        if scenes:
            samples_idx = GNSSParser.samples_idx_from_scenes(scenes=scenes, max_len=len(self.best_gnss_timestamps))
            self.extract_images_samples_idx(samples_idx, cam_folders, self.cams_files[:])

        else: 
            print("")

        # CamParser.extract_images(raw_dataroot=self.raw_dataroot)

    
    def extract_images_samples_idx(self, samples_idx, cam_folders, cams_files): 

        
        for cam_file, cam_folder, (cam_name, cam_timestamps) in zip(cams_files, cam_folders, self.camera_parameters.items()): 
            CamParser.extract_images(cam_file, cam_folder, cam_name, cam_timestamps, samples_idx, self.freq_ratio)


        
    def load_cams_timestamps(self):
        pass

    def create_database(self):
        
        cd_sensors = self.create_calibrated_sensor()
        ego_poses = self.create_ego_pose()


    def create_calibrated_sensor(self):
        cd_sensors = []

        for i, (k, v) in enumerate(self.camera_parameters.items()):
            cd_sensor = CalibratedSensor(translation=v['t'], rotation=transformation_utils.euler_to_quaternion_yaw(v['roll_pitch_yaw']),\
                 cx=v['cx'], cy=v['cy'], fw_coeff=v['fw_coeff'], bw_coeff=v['bw_coeff'], nuscene_camera_intrinsics=v['nuscenes_cam_intrinsics'], nuscene_image_size=v['nuscene_image_size'])
            cd_sensors.append(cd_sensor.__dict__)

        return cd_sensors


    def create_ego_pose(self): 
        ego_poses = []

        x_translation = self.ego_xy_position.x - self.ego_xy_position.x[0]
        y_translation = self.ego_xy_position.y - self.ego_xy_position.y[0]

        for i, gnss_timestamp in enumerate(self.best_gnss_timestamps): 
            ego_poses = EgoPose(translation=[float(x_translation[i]), float(y_translation[i]), float(0)], rotation=0, timestamp=gnss_timestamp)


    def create_samples(self):
        pass


    def create_samples_data(self): 
        pass



if __name__ == "__main__": 


    raw_dataroot = "/cluster/home/terjenf/MapTR/NAP_raw_data"
    trip = "Trip077"
    processed_dataroot = "/cluster/home/terjenf/naplab/data"
    nuscene_path = "/cluster/home/terjenf/naplab/naplab/naplab_parser/metadata_nuscenes/nuscene_metadata.json"

    selected_cams = [
        'C1_front60Single', 
        'C8_R2', 
        'C7_L2',
        'C4_rearCam', 
        'C6_L1',
        'C5_R1']


    naplab_parser = NapLabParser(raw_dataroot=raw_dataroot, trip=trip, processed_dataroot=processed_dataroot, selected_cams=selected_cams)

    naplab_parser.visualize_route(processed_dataroot=processed_dataroot)

    naplab_parser.visualize_route_bearings(processed_dataroot=processed_dataroot, refrence_frame=500)
    naplab_parser.visualize_route_bearings(processed_dataroot=processed_dataroot, refrence_frame=500, scenes=(50,60))
    naplab_parser.visualize_route_bearings(processed_dataroot=processed_dataroot, refrence_frame=500, scenes=(105,131))


    naplab_parser.extract_images(scenes=(105,131))

    print("HE")