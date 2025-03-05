import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable
import pickle

import imageio
import cv2
import matplotlib.pyplot as plt
from PIL import Image

cam_layout = {
        'C1_front60Single': 1,
        'C8_R2': 2, 
        'C7_L2': 0, 
        'C4_rearCam': 4, 
        'C6_L1':  3, 
        'C5_R1':  5,
    }


naplab2nuscenes = {
        'C1_front60Single': 'CAM_FRONT',
        'C8_R2': 'CAM_FRONT_RIGHT', 
        'C7_L2': 'CAM_FRONT_LEFT', 
        'C4_rearCam': 'CAM_BACK', 
        'C6_L1': 'CAM_BACK_LEFT', 
        'C5_R1':  'CAM_BACK_RIGHT'
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

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))


    def plot_sample_canvas(self, cam_dict, gnss_timestamp, ind, save=False, model_name=False): 

        fig, axis = plt.subplots(2,3, figsize=(8,6))
        axis = axis.flatten()

        plt.subplots_adjust(wspace=0.0, hspace=0.0) 
        
        for cam_name, v in cam_dict.items():
            
            axis[cam_layout[cam_name]].imshow(Image.open(v[0]))
            axis[cam_layout[cam_name]].axis('off')
            axis[cam_layout[cam_name]].set_title(naplab2nuscenes[cam_name])
            axis[cam_layout[cam_name]].text(0.5, -0.1, v[1], ha='center', va='top', transform=axis[cam_layout[cam_name]].transAxes, fontsize=12, color='black')

        if save: 
            if model_name:
                file_location = os.path.join(self.dataroot, self.trip, model_name)
            else: 
                file_location = os.path.join(self.dataroot, self.trip)
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            filename = os.path.join(file_location, f"sample_{ind}.png")
            plt.tight_layout()
            fig.suptitle(f"{self.trip} | sample index: {ind} | GNSS timestamp {gnss_timestamp}")
            plt.savefig(filename)
            print("Saved image: ", filename)
        
        else: 
            plt.show()
            plt.clf()  
            plt.close()

    def convert_images_to_video(self, image_files, output_file, fps):
        

        # Read the first image to get its dimensions
        first_image = cv2.imread(image_files[0])
        height, width, _ = first_image.shape

        # Create a VideoWriter object to save the video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

        # Iterate over each image and write it to the video
        for image_file in image_files:
            
            frame = cv2.imread(image_file)
            video.write(frame)
        
        # Release the video writer and close the video file
        video.release()
        cv2.destroyAllWindows()


    def save_video(self, image_list, out_path, name):

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        out_file = os.path.join(out_path, name)


        with imageio.get_writer(out_file, fps=10) as writer:
            for image in image_list:
                writer.append_data(image)


    def save_as_video(self, image_list, mp4_output_path, name, scale=None):
   
        if not os.path.exists(mp4_output_path):
            os.makedirs(mp4_output_path)

        mp4_output_path = os.path.join(mp4_output_path, f"{name}_video.mp4")
            
        images = [Image.fromarray(img).convert("RGBA") for img in image_list]
        if scale is not None:
            w, h = images[0].size
            images = [img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS) for img in images]
        images = [Image.new('RGBA', images[0].size, (255, 255, 255, 255))] + images
        try:
            imageio.mimsave(mp4_output_path, images,  format='MP4',fps=10)
        except ValueError: # in case the shapes are not the same, have to manually adjust
            resized_images = [img.resize(images[0].size, Image.Resampling.LANCZOS) for img in images]
            print('Size not all the same, manually adjust...')
            imageio.mimsave(mp4_output_path, resized_images,  format='MP4',fps=10)
        print("mp4 saved to : ", mp4_output_path)


    def visualize_firste_sample_scene(self, scene_id, save=False):

        scene = self.scene[scene_id]

        sample_token = scene['first_sample_token']

        sample_id = self._token2ind['sample'][sample_token]

        self.visualize_sample(sample_id, save=save)

    def visualize_sample_compact(self, ind, model_name=None, save=True, return_image=True): 


        sample = self.sample[ind]

        data = sample['data']

        gnss_timestamp = sample['timestamp']
      
        cam_dict = {}
        for cam_name, cam_token in data.items():
            sample_data = self.get('sample_data', cam_token)
            filename = sample_data['filename']
            #sample_data_timestamp = sample_data['timestamp']

            cam_dict[cam_name] = [filename]
        

        fig, axis = plt.subplots(2,3, figsize=(6,3))
        axis = axis.flatten()

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        
        for cam_name, v in cam_dict.items():
            
            axis[cam_layout[cam_name]].imshow(Image.open(v[0]))
            axis[cam_layout[cam_name]].axis('off')
            #axis[cam_layout[cam_name]].set_title(cam_name)
            

        if save: 
            file_location = os.path.join(self.dataroot, self.trip, model_name, "plots")
            if not os.path.exists(file_location):
                os.makedirs(file_location)

            filename = os.path.join(file_location, f"sample_{ind}.png")
            plt.tight_layout()
            fig.suptitle(f"{self.trip} | sample index: {ind} ")
            plt.savefig(filename)
            viz_image = imageio.imread(filename)
            plt.clf()  
            plt.close()
            return viz_image

    
        plt.show()
        plt.clf()  
        plt.close()
        return 




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

    naplab.visualize_sample(4290, save=True)

    print("heis")










