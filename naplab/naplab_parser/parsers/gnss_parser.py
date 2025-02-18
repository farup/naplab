import numpy as np 
import cv2 
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
from pyproj import Geod
import sys 


class GNSSParser: 


    @classmethod
    def set_nbr_samples(cls, nbr_samples):
        cls.nbr_samples = nbr_samples
    
    @staticmethod
    def nmea_to_decimal(coord):
        """ Convert MNEA latidue and and longitude to decimal degrees
        Args
            Coords: 
                Latitude is represented as ddmm.mmmm
                longitude is represented as dddmm.mmmm
                - dd or ddd is degrees
                - mm.mmmm is minutes and decimal fractions of minutes
        Returns:
            float: degrees in decimal 
        """
        degrees = int(coord[:2])  # First 2 digits are degrees
        minutes = float(coord[2:])  # Rest is minutes
        decimal_degrees = degrees + (minutes / 60)
        return decimal_degrees
    

    @staticmethod
    def get_ego_xy_positions(lat_lon):
        """
        Extract coordinates in 3857 (in meters for latitude and longitude)
        
        """

        # Create a GeoDataFrame from latitude and longitude
        geometry = [Point(coord[1], coord[0]) for coord in lat_lon]

        gdf = gpd.GeoDataFrame(geometry=geometry)

        gdf.set_crs("EPSG:4326", inplace=True)
        # EPSG:4326 (WGS 84)  represents locations using latitude and longitude (degrees)
        # standard CRS for GPS, Google Earth, OpenStreetMap, and most global datasets.

        gdf = gdf.to_crs(epsg=3857)

        return gdf.get_coordinates()
    
    @staticmethod
    def azimuth_to_bearing(fwd_azimuth):
        if fwd_azimuth < 0:
            fwd_azimuth = (fwd_azimuth + 360) % 360 # convert to 0-360 degrees
        return fwd_azimuth

    @staticmethod
    def compute_bearing(points):
        """ Compute forward azimuths between two GNNS points. 
        Convert to and returns bearings   """

        cur_points = np.array(points)[:-1]
        next_points = np.array(points)[1:]

        geod = Geod(ellps="WGS84")

        fwd_azimuth, back_azimuth, distance = geod.inv(cur_points[:, 1], cur_points[:, 0], next_points[:, 1], next_points[:, 0])

        vec_func = np.vectorize(GNSSParser.azimuth_to_bearing)
        fwd_azimuth = vec_func(fwd_azimuth)

        return fwd_azimuth
    

    
    @staticmethod
    def create_direction_vectors(bearings, start_refrence_point=False, negate=False):

        if start_refrence_point:
            bearings = bearings - bearings[start_refrence_point]

        if negate: 
            bearings = 360  - np.array(bearings) 

        radians = np.deg2rad(bearings)
        hos = np.cos(radians)
        mot = np.sin(radians)

        vectors = np.stack((hos,mot), axis=-1)
        return vectors, bearings


    @staticmethod
    def get_ego_positions_in_meters(lat_lon):
        """
        Extract coordinates in 3857 (in meters for latitude and longitude)
        
        """
        # Create a GeoDataFrame from latitude and longitude
        geometry = [Point(coord[1], coord[0]) for coord in lat_lon]

        gdf = gpd.GeoDataFrame(geometry=geometry)

        gdf.set_crs("EPSG:4326", inplace=True)
        # EPSG:4326 (WGS 84)  represents locations using latitude and longitude (degrees)
        # standard CRS for GPS, Google Earth, OpenStreetMap, and most global datasets.

        gdf = gdf.to_crs(epsg=3857)

        return gdf.get_coordinates()
    

    @staticmethod
    def get_gnns_data(gnss_file):
        """ Extract latitude, longitude, and UNIX timestamps from GPGGA messages from gnss_file.
            Converts latitude and longitude to decimal degree represnation. 

        Args:
            gnss_file: file path to NMEA (National Marine Electronics Association) binary file. 

        Returns:
            lat_lon (list): list of tuples with lat and lon 
            timestamps_gnss (list): list with timestamp
        """
        with open(gnss_file, "rb") as f:
            count = 0

            timestamps_gnss = []

            lat_lon = []
    
            for line in f.readlines():
                if line.startswith(b"$GPGGA"):
                
                    coord = (line.split(b",")[2:6])
                    latitude = float(coord[:2][0].strip())  #  6324.8972646 
                    longitude = float(coord[2:][0].strip()) # 1023.9477304 
                    
                    lat = GNSSParser.nmea_to_decimal(str(latitude)) # 63.41495441
                    lon = GNSSParser.nmea_to_decimal(str(longitude))

                    lat_lon.append([lat, lon])
                    
                    time_stamp = (int(line.split(b" ")[-1].strip()))

                    #dict_obj[time_stamp] = {'x': x, 'y':y}
                    timestamps_gnss.append(time_stamp)
                    
                    count += 1

        print(f"Read and extracted {count} lines of data")
        return lat_lon, timestamps_gnss
    
    @staticmethod
    def samples_idx_from_scenes(scenes, max_len):
        if isinstance(scenes, list): 
            scenes_sample_idx = [scene * GNSSParser.nbr_samples for scene in scenes] # each scene has (40) samples 
            if (max(scenes_sample_idx)+ GNSSParser.nbr_samples) > max_len: 
                raise ValueError("Scenes not in data")

            indx_list = []
            for scene_sample_idx in scenes_sample_idx:

                end_scene_idx =+ (scene_sample_idx + GNSSParser.nbr_samples)


                indx_list = np.arange(scene_sample_idx, end_scene_idx) 
                indx_list.append(indx_list)
            
            return np.array(indx_list)

             
        elif isinstance(scenes, tuple):
            scenes_sample_idx = [scene * GNSSParser.nbr_samples for scene in scenes]
            if (max(scenes_sample_idx) + GNSSParser.nbr_samples) > max_len:
                raise ValueError("Scenes not in data")
            
            return np.arange(scenes_sample_idx[0], scenes_sample_idx[1])
            

        elif isinstance(scenes, int):
            scenes_sample_idx = scenes * GNSSParser.nbr_samples
            if ((scenes_sample_idx) + GNSSParser.nbr_samples) > max_len: 
                raise ValueError("Scenes not in data")

            end_scene_idx =+ (scenes_sample_idx + GNSSParser.nbr_samples)

            return np.arange(scenes_sample_idx, end_scene_idx)

          
    @staticmethod
    def plot_route_bearings(bearings, ego_xy_positions, freq_ratio, refrence_frame=0, scenes=False, save=False, processed_dataroot=None):

        x_tranlsation = ego_xy_positions.x - ego_xy_positions.x[0]
        y_tranlsation = ego_xy_positions.y - ego_xy_positions.y[0]

        plt.figure(figsize=(10,15))
        plt.scatter(x_tranlsation[::freq_ratio], y_tranlsation[::freq_ratio])

        if refrence_frame:
            vectors, _ = GNSSParser.create_direction_vectors(bearings, refrence_frame)
            plt.scatter(x_tranlsation[refrence_frame], y_tranlsation[refrence_frame], s=200, label="Refrence Point For Relative Direction" )

        else: 
            vectors, _ = GNSSParser.create_direction_vectors(bearings)
            plt.scatter(x_tranlsation[refrence_frame], y_tranlsation[refrence_frame], s=200, label="Refrence Point For Relative Direction" )
        
        plt.quiver(x_tranlsation[1::20], y_tranlsation[1::20], vectors[::20, 1], vectors[::20, 0])

        if scenes:
            sample_idxs = GNSSParser.samples_idx_from_scenes(scenes=scenes, max_len=len(x_tranlsation))
            
            ego_scenes_x = x_tranlsation[sample_idxs]
            ego_scenes_y = y_tranlsation[sample_idxs]


            plt.scatter(ego_scenes_x[::freq_ratio], ego_scenes_y[::freq_ratio], 5, color="red", label="Selected Scenes")
        
        plt.xlabel("Meters")
        plt.ylabel("Meters")
        plt.title("Positions with Direction")
        plt.legend()

        if save:
            if scenes:
                filename = f'route_with_bearings_ref_frame{refrence_frame}_{str(scenes)}.png'
            else:
                filename = f'route_with_bearings_ref_frame{refrence_frame}.png'

            path = os.path.join(processed_dataroot, "plots")
            if not os.path.exists(path): 
                os.makedirs(path)

            plt.savefig(os.path.join(path, filename))
            print("Plot saved ", os.path.join(path,filename)) 

    
    @staticmethod
    def plot_route(lat_lon, processed_dataroot=False):
        # Create a GeoDataFrame from latitude and longitude
        geometry = [Point(coord[1], coord[0]) for coord in lat_lon]

        gdf = gpd.GeoDataFrame(geometry=geometry)

        gdf.set_crs("EPSG:4326", inplace=True)
        # EPSG:4326 (WGS 84)  represents locations using latitude and longitude (degrees)
        # standard CRS for GPS, Google Earth, OpenStreetMap, and most global datasets.

        gdf = gdf.to_crs(epsg=3857)
        # projected coordinate system that represents locations in meters.
        # Convert the coordinates to a CRS that works with contextily basemaps (Web Mercator)
        fig, ax = plt.subplots(figsize=(10, 10))  # Adjust width and height here
        # Plot the map with reference basemap

        gdf.plot(marker="o", color="blue", markersize=1, ax=ax)
        #ax.set_xlim(-800, 800)

        # Add reference map (OpenStreetMap)
        ctx.add_basemap(ax, crs=gdf.crs.to_string())

        start = gdf.get_coordinates().iloc[0]
        end = gdf.get_coordinates().iloc[-1]

        plt.scatter(start.x, start.y, label="Start", color="red")
        plt.scatter(end.x, end.y, label="End", color="black")


        # Add labels, title, and other customizations
        plt.title("Latitude/Longitude on Map")
        plt.xlabel("Longitude in meters")
        plt.ylabel("Latitude in meters")
        plt.legend()

        if processed_dataroot: 
            path = os.path.join(processed_dataroot, "plots")
            if not os.path.exists(path): 
                os.makedirs(path)

            plt.savefig(os.path.join(path,'route.png'))
            print("Plot saved ", os.path.join(path,'route.png'))

