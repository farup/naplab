@PIPELINES.register_module(force=True)
class ResizeMultiViewImagesNapLab(object):
    """Resize mulit-view images and change intrinsics
    If set `change_intrinsics=True`, key 'cam_intrinsics' and 'ego2img' will be changed

    Args:
        size (tuple, optional): resize target size, (h, w).
        change_intrinsics (bool): whether to update intrinsics.
    """
    def __init__(self, size=None, scale=None, change_intrinsics=True):
        self.size = size
        self.scale = scale
        assert size is None or scale is None
        self.change_intrinsics = change_intrinsics

    def __call__(self, results:dict):

        new_imgs, post_intrinsics, post_ego2imgs, post_ego2cams, cx_s, cy_s, scaleH_s, scaleW_s, fw_coeff_s = [], [], [], [], [], [], [], [], []

        for img,  cam_intrinsic, ego2img, ego2cam, cx, cy, fw_coeff in zip(results['img'], results['cam_intrinsics'], results['ego2img'], results['ego2cam'], results['cx'], results['cy'], results['fw_coeff']):
            if self.scale is not None:
                h, w = img.shape[:2]
                target_h = int(h * self.scale)
                target_w = int(w * self.scale)
            else:
                target_h = self.size[0]
                target_w = self.size[1]
            
            tmp, scaleW, scaleH = mmcv.imresize(img,
                                                # NOTE: mmcv.imresize expect (w, h) shape
                                                (target_w, target_h),
                                                return_scale=True)
            new_imgs.append(tmp)

            rot_resize_matrix = np.array([
                [scaleW, 0,      0,    0],
                [0,      scaleH, 0,    0],
                [0,      0,      1,    0],
                [0,      0,      0,    1]])
            post_intrinsic = rot_resize_matrix[:3, :3] @ cam_intrinsic
            post_ego2img = rot_resize_matrix @ ego2img
            post_ego2cam = rot_resize_matrix @ ego2cam
            post_intrinsics.append(post_intrinsic)
            post_ego2imgs.append(post_ego2img)
            post_ego2cams.append(post_ego2cam)

            cx_new = cx * scaleW
            cy_new = cy * scaleH

            cx_s.append(cx_new)
            cy_s.append(cy_new)

            fw_coeff_s.append(np.array(fw_coeff) * (np.sqrt(scaleW * scaleH)))

            scaleH_s.append(scaleH)
            scaleW_s.append(scaleW)


        results['img'] = new_imgs
        results['img_shape'] = [img.shape for img in new_imgs]
        if self.change_intrinsics:
            results.update({
                'cam_intrinsics': post_intrinsics,
                'ego2img': post_ego2imgs,
                'ego2cam': post_ego2cams, 
                'cx': cx_s,
                'cy': cy_s,
                'fw_coeff': fw_coeff_s, 
                'scaleH': scaleH_s,
                'scaleW': scaleW_s
            })

        return results
