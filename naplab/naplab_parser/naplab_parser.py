import os
import numpy as np

from naplab.naplab_parser.parsers.gnss_parser import GNSSParser
from naplab.naplab_parser.parsers.cam_parser import CamParser
import errno

from naplab.naplab_parser.utils import parsing_utils, timestamps_utils, transformation_utils
from naplab.naplab_parser.tables import Scene, Sample, SampleData, CalibratedSensor, EgoPose, Map



class NapLabParser: 

    def __init__(self, datafolder, trip, save_path, selected_cams, nuscnes_path=False):

        if not os.path.exists(dataroot := os.path.join(datafolder, trip)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataroot)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.dataroot = dataroot
        self.save_path = save_path
        self.selected_cams = selected_cams
        self.start_cam_index = None
        self.end_gnss_index = None
        self.nuscnes_path = nuscnes_path

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

        lat_lon = self.lat_lon[2000]
        lat_lon2 = self.lat_lon[2001]

        self.bearings = GNSSParser.compute_bearing(self.lat_lon)
        

        print("TER")


    def load_paths(self):

        self.absoulte_files = parsing_utils.get_subfolders(self.dataroot)
        self.camera_parameters_file = parsing_utils.get_files(self.absoulte_files, file_format="json", file_key="camera")
        self.gnss_file = parsing_utils.get_files(self.absoulte_files, file_format="bin", file_key="gnss50")
        self.cams_timestamps_files = parsing_utils.get_filetypes(self.absoulte_files, file_format="timestamps")
    
    def set_start_stop_indexes(self, cam_name): 

        print("Updating start cams and end gnss timestamps...")
        self.start_cam_index = self.res[cam_name]['arg1_index_start']
        self.end_gnss_index = self.res[cam_name]['arg2_index_end']

        self.adjust_timestamps()

    def get_average_sync_diff(self, cams_timestamps, gnss_timestamps): 

        results = []
        for cam_name, cam_timestamps in cams_timestamps.items():
            freq_ration = round(timestamps_utils.get_freq_ration(cam_timestamps,gnss_timestamps ))
            sec = timestamps_utils.get_sync_diff(cam_timestamps[::freq_ration], gnss_timestamps)
            results.append(sec)

        average_sec_diff = np.mean(np.abs(np.array(results)), axis=-1)

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


    def visualize_route(self):

        GNSSParser.plot_route(self.lat_lon[:], save_path=self.dataroot)

    def visualize_route_bearings(self, start_refrence_point=False): 

        vectors, _ = GNSSParser.create_direction_vectors(self.bearings[:])

        GNSSParser.plot_route(self.lat_lon[:], save_path=self.dataroot, vectors=vectors)
    
    def visualize_sync_diffs(self, save=False):

        timestamps_utils.plot_sync_diff(self.res, save=save)


    def extract_images(self, scenes=False):
        
        CamParser.extract_images()

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


    datafolder = "/cluster/home/terjenf/MapTR/NAP_raw_data"
    trip = "Trip077"
    save_path = "/cluster/home/terjenf/naplab/data"
    nuscene_path = "/cluster/home/terjenf/naplab/naplab/naplab_parser/metadata_nuscenes/nuscene_metadata.json"

    selected_cams = [
        'C1_front60Single', 
        'C8_R2', 
        'C7_L2',
        'C4_rearCam', 
        'C6_L1',
        'C5_R1']


    naplab_parser = NapLabParser(datafolder=datafolder, trip=trip, save_path=save_path, selected_cams=selected_cams)

    



    print("Bru")