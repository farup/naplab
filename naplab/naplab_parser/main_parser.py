import os


from naplab.naplab_parser.parsers.gnss_parser import GNSSParser
from naplab.naplab_parser.parsers.cam_parser import CamParser
import errno

from naplab.naplab_parser.utils import parsing_utils, timestamps_utils




class NapLabParser: 

    def __init__(self, datafolder, trip, save_path):


        if not os.path.exists(dataroot := os.path.join(datafolder, trip)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), dataroot)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.dataroot = dataroot
        self.save_path = save_path

        self.load_paths()
        self.cams_timestamps = parsing_utils.get_cams_timestamps(self.cams_timestamps_files)

        self.lat_lon, self.gnss_timestamps = GNSSParser.get_gnns_data(self.gnss_file)

    
    def load_paths(self):

        self.absoulte_files = parsing_utils.get_subfolders()
        self.gnss_file = parsing_utils.get_files(self.absoulte_files, file_format="bin", file_key="gnss50")
        self.cams_timestamps_files = parsing_utils.get_filetypes(self.absoulte_files, file_format="timestamps")
    

    def load_cams_timestamps():



    def parse_trip(self, trip): 
        pass





if __name__ == "__main__": 
    print("Bru")