import os 

import numpy as np

from . import parsing_utils

import matplotlib.pyplot as plt

# from ..parsers.gnss_parser import GNSSParser

from naplab.naplab_parser.parsers.gnss_parser import GNSSParser



def get_avgerage_freq(timestamps): 

    microsec_diffs = np.mean(np.abs(np.array(timestamps[:-1]).astype(int) - np.array(timestamps[1:]).astype(int)))

    return 1 / (microsec_diffs / 1e6)

def get_freq_ration(timestamps1, timestamps2):

    freq1 = round(get_avgerage_freq(timestamps1))
    freq2 = round(get_avgerage_freq(timestamps2))

    return freq1 / freq2


def get_sync_diff(timestamps1, timestamps2):

    assert len(timestamps1) == len(timestamps2), f" arg1 {len(timestamps1)} should be equal arg2 {len(timestamps2)}"
    
    np_timestamps1 = np.array(timestamps1).astype(int)
    np_timestamps2 = np.array(timestamps2).astype(int)

    seconds = (np.mean(np.abs(np_timestamps1 - np_timestamps2))) / 1e6
    # print(f"Average time (in seconds) between timestamps: {seconds}")
    return seconds


def preparare_timestsamps(cam_timestamps):
    """
    Some camera timestamps list are longer than other camera timestamps list for the same trip. 
    Cut out the end last time stamp 

    args: 
        cam_timestamps (dict)
    
    """

    lens = [len(timestamps) for timestamps in cam_timestamps.values()]
    min_len = min(lens)
    
    for cam_name, timestamp in cam_timestamps.items(): 
        if len(timestamp) > min_len:
            cam_timestamps[cam_name] = (timestamp[:-(len(timestamp) - min_len)]) 

    return cam_timestamps
 

def get_best_syncs(cams_timestamps, gnss_timestamps):
    """ Calculate the sync alignment in microsconds with different start indexes"""

    print("Starting Best Sync Calculations...")
    res = {}
    for cam_name, cam_timestamps in cams_timestamps.items(): 
        freq_ratio = round(get_freq_ration(cam_timestamps, gnss_timestamps))
        res[cam_name] = get_best_start_stop_index(cam_timestamps, gnss_timestamps, freq_ratio)

    return res


def get_best_start_stop_index(file_1_timestamps, file_2_timestamps, freq_ratio, runs=20):
    """
    Iterates through different starting poistions for the longest list, 
    and corresponding end positions for the shortes list. 
    Finds the best section => camera start index and gnss end index 
    
    
    """


    assert len(file_1_timestamps[::freq_ratio]) >= len(file_2_timestamps), f"Assume arg1 can be sampled every {freq_ratio}, due to higher freq "
    dict_best = {'best_score': 1000000000}
    
 
    seconds_list = []
    arg1_index_starts = []
    arg2_index_ends = []

    for i in range(runs):
        if i == 0:
            #print("Start")
            seconds = get_sync_diff(file_1_timestamps[::freq_ratio], file_2_timestamps)
            if seconds < dict_best['best_score']:
                dict_best['best_score'] = seconds 
                dict_best['arg1_index_start'] = 0
                dict_best['arg2_index_end'] = 0
                dict_best['arg1_index_start_timestamp'] = file_1_timestamps[i]
                dict_best['arg2_index_end_timestamp'] = file_1_timestamps[i]
            seconds_list.append(seconds)
            arg1_index_starts.append(0)
            arg2_index_ends.append(0)

            arg1_index_start =  3*(i+1)
            arg2_index_end = -1

        else:
            arg1_index_start =  3*(i+1) + 1
            arg2_index_end = -i-2

        # print(f"\nArgument1 start indx: {arg1_index_start}, Argument2 end index: {arg2_index_end}")

        seconds = get_sync_diff(file_1_timestamps[arg1_index_start::freq_ratio], file_2_timestamps[:arg2_index_end])
        if seconds < dict_best['best_score']:
            dict_best['best_score'] = seconds 
            dict_best['arg1_index_start'] = arg1_index_start
            dict_best['arg2_index_end'] = arg2_index_end
            dict_best['arg1_index_start_timestamp'] = file_1_timestamps[i]
            dict_best['arg2_index_end_timestamp'] = file_1_timestamps[i]

        seconds_list.append(seconds)
        arg1_index_starts.append(arg1_index_start)
        arg2_index_ends.append(arg2_index_end)
                
    dict_best['seconds_list']  = seconds_list
    dict_best['arg1_index_starts']  = arg1_index_starts
    dict_best['arg2_index_ends']  = arg2_index_ends


    return dict_best



def plot_sync_diff(res, save=False):
    fig, axis = plt.subplots(1,2, figsize=(10,10)) 


    for cam_name, v in res.items(): 

        if cam_name == "C1_front60Single":
            axis[0].scatter(v['arg1_index_start'], v['best_score'], color='red', label=f"Best {cam_name}", marker='o')
            axis[1].scatter(v['arg2_index_end'], v['best_score'], color='red',label=f"Best {cam_name}",  marker='o')

        axis[0].plot(v['arg1_index_starts'], v['seconds_list'], label=f"{cam_name}")
        axis[1].plot(v['arg2_index_ends'], v['seconds_list'], label=f"{cam_name}")
    
    axis[0].set_title("Cam")
    axis[0].set_xlabel("Cam Timstamp Start Index")
    axis[0].set_ylabel("Seconds")

    axis[1].set_title("GNSS")
    axis[1].set_xlabel("GNNS Timestamp End Index")
    axis[1].set_ylabel("Seconds")



    fig.suptitle("Average Synchronisation Time Difference")
    
    plt.tight_layout()
    plt.legend()
   

    if save: 
        plt.savefig("./sync_diff.png")
    else:
        plt.show()
    
    plt.clf()

        


if __name__ == "__main__": 


    root_folder = "/cluster/home/terjenf/MapTR/NAP_raw_data/"
    trip = "Trip077"

    subfolders = parsing_utils.get_subfolders(root_folder)
    subfolders = parsing_utils.get_subfolders(os.path.join(root_folder, trip))
    timestamps_files = parsing_utils.get_filetypes(subfolders, file_format="timestamps")
    timestamps = parsing_utils.get_timestamps(timestamps_files[0])
    cams_timestamps = parsing_utils.get_cams_timestamps(timestamps_files)


    cams_timestamps_ = preparare_timestsamps(cams_timestamps)

    gnss_files = parsing_utils.get_files(subfolders, file_format="bin", file_key="gnss50")

    lat_lon, timestamps_gnss = GNSSParser.get_gnns_data(gnss_files[0])


    prepared_cam_timestamps = preparare_timestsamps(cams_timestamps_)


    freq_1 = get_avgerage_freq(timestamps)
    freq_2 = get_avgerage_freq(timestamps_gnss)


    res = get_best_syncs(prepared_cam_timestamps, timestamps_gnss, round(freq_1/freq_2))


    plot_sync_diff(res, save=True)

 


    print("Heis")






