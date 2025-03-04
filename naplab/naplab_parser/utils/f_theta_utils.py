import numpy as np
import matplotlib.pyplot as plt
import os

def poly_func(r, float_bw):
    res = np.sum([j*r**i for i, j in enumerate(float_bw)])
    return res


def poly_easy(r, float_bw):
    sum_v = 0
    for i, coeff in enumerate(float_bw): 
        sum_v += coeff*r**i

    return sum_v



def get_fw_coeff(w, bw_coeff, return_all=False):
    """ Fit forward coefficients to values
    computed from the backward transform """

    r_distances = np.linspace(0, w/2, 500)

   

    thetas = [poly_func(r, bw_coeff) for r in  r_distances]

    forward_degree = len(bw_coeff) - 1

    fw_coeff = np.polyfit(thetas, r_distances, forward_degree)

    fw_coeff[0] = 0
    
    fw_coeff_list = [float(f) for f in fw_coeff]

    if return_all: 
        return fw_coeff_list, thetas, r_distances


    return fw_coeff_list


def get_fw_coeff_start_sigle_0(w, bw_coeff, return_all=False):
    """ Fit forward coefficients to values
    computed from the backward transform """

    r_distances = np.linspace(0, w, 500)


    thetas = [poly_func(r, bw_coeff) for r in  r_distances]

    
    #thetas_2 = [poly_easy(r, bw_coeff) for r in r_distances]


    forward_degree = len(bw_coeff) - 1


    fw_coeff = np.polyfit(thetas, r_distances, forward_degree)

    
    fw_coeff_list = [float(f) for f in fw_coeff]

    fw_coeff_list.append(0)
    if return_all: 
        
        return fw_coeff_list, thetas, r_distances

    
    fw_coeff_list.reverse()

    return fw_coeff_list


def get_fw_coeff_start_0(w, bw_coeff, return_all=False):
    """ Fit forward coefficients to values
    computed from the backward transform """

    r_distances = np.linspace(0, w, 500)


    thetas = [poly_func(r, bw_coeff) for r in  r_distances]

    
    #thetas_2 = [poly_easy(r, bw_coeff) for r in r_distances]


    forward_degree = len(bw_coeff) - 1


    fw_coeff = np.polyfit(thetas, r_distances, forward_degree)

    
    fw_coeff_list = [float(f) for f in fw_coeff]

    fw_coeff_list.append(0)
    fw_coeff_list.append(0)
    if return_all: 
        
        return fw_coeff_list, thetas, r_distances

    
    fw_coeff_list.reverse()

    return fw_coeff_list




def plot_forward_only(fw_coeff_list, thetas, r, output_path=False, cam_name="naplab", save=False): 

    fig, axis = plt.subplots(1, figsize=(6,10))

    axis.scatter(np.rad2deg(thetas), r)
    axis.set_ylabel("r distances")
    axis.set_xlabel("theta angles in degrees")
    axis.set_xlim(0, 90)
    axis.set_ylim(0, 1000)

    r_comp = np.polyval(fw_coeff_list, thetas)

    axis.plot(np.rad2deg(thetas), r_comp, label="Forward Function", color='r')

    plt.legend()
    #plt.suptitle("Forward Transformation")

    if save: 
        file_location = os.path.join(output_path, "plots")
        if not os.path.exists(file_location):
            os.makedirs(file_location)

        filename = os.path.join(file_location, f"forward_func_{cam_name}_{len(fw_coeff_list)}.png")

        plt.tight_layout()
        plt.savefig(filename)
        print("Saved fig:", filename)
        plt.clf()  
        plt.close()




def plot_forward(fw_coeff_list, thetas, r, output_path=False, cam_name="naplab", save=False): 

    fig, axis = plt.subplots(1,2, figsize=(10,10))

    axis[0].scatter(r, np.rad2deg(thetas))
    axis[0].set_xlabel("r distances")
    axis[0].set_ylabel("theta angles in degrees")

    axis[1].scatter(np.rad2deg(thetas), r)
    axis[1].set_ylabel("r distances")
    axis[1].set_xlabel("theta angles in degrees")
    axis[1].set_xlim(0, 90)
    axis[1].set_ylim(0, 1500)

    r_comp = np.polyval(fw_coeff_list, thetas)

    axis[1].plot(np.rad2deg(thetas), r_comp, label="Forward Function", color='r')

    plt.legend()
    #plt.suptitle("Forward Transformation")

    if save: 
        file_location = os.path.join(output_path, "plots")
        if not os.path.exists(file_location):
            os.makedirs(file_location)

        filename = os.path.join(file_location, f"forward_transformation_{cam_name}_{len(fw_coeff_list)}.png")

        plt.tight_layout()
        plt.savefig(filename)
        print("Saved fig:", filename)
        plt.clf()  
        plt.close()



    

