import os 
import errno
import json
import numpy as np



def save_tables_json(processed_dataroot, table, filename):

    processed_dataroot_folder = os.path.join(processed_dataroot, "tables")

    if not os.path.exists(processed_dataroot_folder):
        os.makedirs(processed_dataroot_folder)

    new_file = os.path.join(processed_dataroot_folder, filename)
    with open(new_file, 'w') as f: 
        json.dump(table, f, indent=4)
        print("Saved to", new_file)





def create_cam_folders(path, cam_names):

    new_folders = []
    for cam_name in cam_names: 
        new_folder = os.path.join(path, cam_name)
        if not os.path.exists(new_folder): 
            os.makedirs(new_folder)
            print("Created Folder", new_folder)
            new_folders.append(new_folder)
        else:
            new_folders.append(new_folder)
        
    return new_folders 


def get_subfolders(root_folder): 
    """Return subfolders/files of a folder """
    if not os.path.exists(root_folder): 
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), root_folder)

    folders = [os.path.join(root_folder,f) for f in os.listdir(root_folder)]
    folders.sort(key=sort_func)
    return folders



def sort_func(e): 
    """ Used to sort files """
    return e.split("/")[-1]



def get_filetypes(files, file_format): 
    """
    Return files with  of the file_format
    
    args: 
        files (list): files to search in
        file_format (str): file format
    
    """
    selected_files = []
    for file in files:
        if file.split('.')[-1] == file_format: 
            selected_files.append(file)
            
    return selected_files

def get_files(files, file_format, file_key=False, selected_cams=False):
    """ 
    Return files with file_formt (e.g. .bin) and  file_key in filename 
    
    """

    selected_filestypes = get_filetypes(files, file_format)
    
    if file_key:
        selected_files = [] 
        for selected_filestype in selected_filestypes: 
            if file_key in selected_filestype: 
                selected_files.append(selected_filestype)
        return selected_files
    

    if selected_cams: 
        selected_files = [] 
        for selected_filestype in selected_filestypes: 
            if selected_filestype.split("/")[-1].split(".")[0] in selected_cams:
                selected_files.append(selected_filestype)
        return selected_files


    return selected_filestypes


def get_timestamps(timestamp_file): 
    """
    Return list of timestamps
    
    """
    count = 0
    timestamps = []
    with open(timestamp_file, 'r') as file: 
        for timestamp in file: 
            # print(timestamp)
            # break
            timestamps.append(timestamp.split("\t")[-1].strip())
            count += 1
        print(f"Counted {count} timestamps in file {timestamp_file}")
    return timestamps


def get_cams_timestamps(timestamps_files): 

    cam_names = [timstamp_file.split("/")[-1].split(".")[0] for timstamp_file in  timestamps_files]
    cams_timestamps = {}

    for cam_name, timestamps_file in zip(cam_names, timestamps_files): 
        cams_timestamps[cam_name] = get_timestamps(timestamps_file)

    return cams_timestamps





if __name__ == "__main__": 


    root_folder = "/cluster/home/terjenf/MapTR/NAP_raw_data/"
    trip = "Trip077"

    subfolders = get_subfolders(root_folder)

    subfolders = get_subfolders(os.path.join(root_folder, trip))

    timestamps_files = get_filetypes(subfolders, file_format="timestamps")

    cams_timestamps = get_cams_timestamps(timestamps_files)

    cams_files = get_filetypes(subfolders, file_format="h264")

    timestamps = get_timestamps(timestamps_files[0])

    print("Heis")

