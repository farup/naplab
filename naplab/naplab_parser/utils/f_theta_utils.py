import numpy as np



def poly_func(r, float_bw):
    res = np.sum([j*r**i for i, j in enumerate(float_bw)])
    return res



def get_fw_coeff(w, bw_coeff):
    """ Fit forward coefficients to values
    computed from the backward transform """

    r = np.linspace(0, w/2, 500)

    print(f"Number of distances (points) used to calculate fw_coeff: {len(r)}")

    thetas = [poly_func(r, bw_coeff) for r in  r]
    forward_degree = len(bw_coeff) - 1

    fw_coeff = np.polyfit(thetas,r, forward_degree)

    fw_coeff[0] = 0
    
    fw_coeff_list = [float(f) for f in fw_coeff]

    print(f"FW: {fw_coeff}\n")
    print(f"BW: {bw_coeff}\n")

    return fw_coeff_list


