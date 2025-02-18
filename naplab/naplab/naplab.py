import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable
import pickle

import matplotlib.pyplot as plt
from PIL import Image


cam_layout = {
        'C1_front60Single': 1,
        'C8_R2': 2, 
        'C7_L2': 0, 
        'C4_rearCam': 4, 
        'C6_L1':  5, 
        'C5_R1':  3,
    }


class NapLab:
    """
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self, dataroot: str, trip: str, verbose: bool =True):

        self.dataroot = dataroot
        self.trip = trip

        self.table_names = ['calibrated_sensor',
                            'ego_pose', 'sample', 'sample_data', 'scene', 'naplab_map']


        start_time = time.time()

        self.calibrated_sensor = self.__load_table__('calibrated_sensors')
        self.ego_pose = self.__load_table__('ego_poses')

        self.scene = self.__load_table__('scenes')
        self.sample = self.__load_table__('samples')
        self.sample_data = self.__load_table__('samples_data')
        self.naplab_map = self.__load_table__('naplab_map')


        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant trip. """
        return osp.join(self.dataroot, self.trip, "tables")


    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(os.path.join(self.table_root, f"{table_name}.json"), 'rb') as f:
            table = json.load(f)
        return table


    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """
        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # for record in self.sample_data:
        #     if record['is_key_frame']:
        #         sample_record = self.get('sample', record['sample_token'])
        #         sample_record['data'][record['channel']] = record['token']

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    #  ind 2023: [Errno 2] No such file or directory: '/cluster/home/terjenf/NAPLab_car/data/Trip077/samples/C1_front60Single/C1_front60Single_1684740124055143'
    # ind 4023 '/cluster/home/terjenf/NAPLab_car/data/Trip077/samples/C1_front60Single/C1_front60Single_1684740324055064'
    # 4425 FileNotFoundError: [Errno 2] No such file or directory: '/cluster/home/terjenf/NAPLab_car/data/Trip077/samples/C1_front60Single/C1_front60Single_1684740363555034'
    def plot_sample_canvas(self, cam_dict, gnss_timestamp, ind, save): 

        fig, axis = plt.subplots(2,3, figsize=(10,10))
        axis = axis.flatten()
        
        for cam_name, v in cam_dict.items():
            
            axis[cam_layout[cam_name]].imshow(Image.open(v[0]))
            axis[cam_layout[cam_name]].axis('off')
            axis[cam_layout[cam_name]].set_title(cam_name)
            axis[cam_layout[cam_name]].text(0.5, -0.1, v[1], ha='center', va='top', transform=axis[cam_layout[cam_name]].transAxes, fontsize=12, color='black')
        if save: 
            file_location = os.path.join(self.dataroot, self.trip, "plots")
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            filename = os.path.join(file_location, f"sample_{ind}.png")
            plt.tight_layout()
            fig.suptitle(f"{self.trip} | sample index: {ind} | GNSS timestamp {gnss_timestamp}")
            plt.savefig(filename)
        
        else: 
            plt.show()
            plt.clf()  
            plt.close()


    def visualize_firste_sample_scene(self, scene_id, save=False):

        scene = self.scene[scene_id]

        sample_token = scene['first_sample_token']

        sample_id = self._token2ind['sample'][sample_token]

        self.visualize_sample(sample_id, save=save)


    def visualize_sample(self, ind, save=False):

        sample = self.sample[ind]

        data = sample['data']

        gnss_timestamp = sample['timestamp']
      
        cam_dict = {}
        for cam_name, cam_token in data.items():
            sample_data = self.get('sample_data', cam_token)
            filename = sample_data['filename']
            sample_data_timestamp = sample_data['timestamp']

            cam_dict[cam_name] = [filename, sample_data_timestamp]

        self.plot_sample_canvas(cam_dict, gnss_timestamp, ind, save)
        

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]
    

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]


    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]


if __name__ == "__main__": 

    dataroot ="/cluster/home/terjenf/naplab/data"
    trip="Trip077"

    naplab = NapLab(dataroot=dataroot, trip=trip)


    naplab.visualize_firste_sample_scene(50, save=True)

    naplab.visualize_sample(2000, save=True)


 

    #naplab.visualize_sample(600, save=True)

    print("heis")










