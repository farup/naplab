


def get_cam_props(cam_name, cam_param):
    w = cam_parm[cam_name]['width']
    h = cam_parm[cam_name]['height']
    cx = cam_parm[cam_name]['cx']
    cy = cam_parm[cam_name]['cy']
    bw_coeff = cam_parm[cam_name]['float_bw']
   
    return w, h, round(cx), round(cy), float_bw


def poly_func(r, float_bw):
    res = np.sum([j*r**i for i, j in enumerate(float_bw)])
    return res



def get_fw_coeff(w, bw_coeff):
    """ Fit forward coefficients to values
    computed from the backward transform """

    r = np.linspace(0, w/2, 500)

    print(f"Number of distances (points) used to calculate: {len(r)}")

    thetas = [poly(r, bw_coeff) for r in  r]
    forward_degree = len(bw_coeff) - 1

    fw_coeff = np.polyfit(thetas,r, forward_degree)

    fw_coeff[0] = 0
    
    fw_coeff_list = [float(f) for f in fw_coeff]

    print(f"BW: {bw_coeff}, FW: {fw_coeff}")

    return fw_coeff_list


