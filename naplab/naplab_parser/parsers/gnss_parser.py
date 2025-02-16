import numpy as np 
import cv2 
import json
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

import sys 


class GNSSParser: 


    @staticmethod
    def parse():
        pass

    

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
    

    def compute_bearing(lat1, lon1, lat2, lon2):
        """
        Compute yaw (bearing) from point (lat1, lon1) to (lat2, lon2).
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        d_lon = lon2 - lon1
        x = np.sin(d_lon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(d_lon)
        
        yaw = np.arctan2(x, y)  # Bearing in radians

        yaw_deg = (np.degrees(yaw) + 360) % 360  # Convert to degrees
        
        return yaw_deg
    

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

                    lat_lon.append((lat, lon))

                    
                    time_stamp = (int(line.split(b" ")[-1].strip()))

                    #dict_obj[time_stamp] = {'x': x, 'y':y}
                    timestamps_gnss.append(time_stamp)
                    
                    count += 1

        print(f"Read and extracted {count} lines of data")
        return lat_lon, timestamps_gnss
    

    @staticmethod
    def plot_route(lat_lon, save_path=False):
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
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()

        if save_path: 
            plt.savefig(os.path.join(save_path,'route.png'))

        


    



    

    
