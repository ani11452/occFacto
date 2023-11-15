from functools import partial 
import open3d as o3d
from argparse import ArgumentParser
from visualizer_pkl import visualize_pkl
from utils import *
import os
import pickle
import numpy as np
from einops import rearrange

def normalize_point_clouds(pc):
    shift = pc.mean(axis=0).reshape(1, 3)
    scale = pc.flatten().std().reshape(1, 1)
    return (pc - shift) / scale

def visualize_npy(out):
    
    id = 0
    color_id = -1
    n = out.shape[0]
    print(n)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(out[id])
    pcd.colors = o3d.utility.Vector3dVector(normalize(out[id]))
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=True) #works for me with False, on some systems needs to be true
    vis.add_geometry(pcd)
    view_contrl = vis.get_view_control()
    view_contrl.rotate(*view_dict['top'])
    view_params = view_contrl.convert_to_pinhole_camera_parameters()
    ren = vis.get_render_option()
    ren.background_color=np.array([1.,1.,1.])
    ren.show_coordinate_frame = True
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

    def incrementid(_vis):
        nonlocal id
        id  += 1
        id %= n
        update_vis(_vis)
    
    def decrementid(_vis):
        nonlocal id
        id -= 1
        id %= n
        update_vis(_vis)
        
    def save_view(_vis):
        nonlocal view_params
        view_contrl = _vis.get_view_control()
        view_params = view_contrl.convert_to_pinhole_camera_parameters()
    
    def update_vis(_vis):
        save_view(_vis)
        _vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(out[id])
        if color_id == -1:
            pcd.colors = o3d.utility.Vector3dVector(normalize(out[id]))
        else:
            pcd.colors = o3d.utility.Vector3dVector(normalize(out[color_id]))
        vis.add_geometry(pcd)
        view_contrl = _vis.get_view_control()
        view_contrl.convert_from_pinhole_camera_parameters(view_params)
        ren = _vis.get_render_option()
        ren.background_color=np.array([1.,1.,1.])
        ren.show_coordinate_frame = True
        _vis.update_geometry(pcd)
        _vis.poll_events()
        _vis.update_renderer()
        print(f'Changed to model id = {id}.')
        return False

                
    def fixColor(_vis):
        nonlocal color_id 
        if color_id != -1:
            color_id = -1
        else:
            color_id = id 
        update_vis(_vis)
   
    
   
    vis.register_key_callback(ord("S"), partial(capture_screen_shot))
    vis.register_key_callback(ord("E"), partial(incrementid))
    vis.register_key_callback(ord("Q"), partial(decrementid))
    vis.register_key_callback(ord("C"), partial(fixColor))

    vis.run()
    vis.destroy_window()




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('pc_dir', type=str, help="path to the point cloud you wish to get the color of.")
    parser.add_argument('--params', action='store_true', help="path to the point cloud you wish to get the color of.")
    
    args = parser.parse_args()
    from scipy.spatial.transform import Rotation
    if args.pc_dir[-3:] == 'npz':
        out = np.load(args.pc_dir)
        out = dict(out)
        visualize_pkl(out, 0, args.params)
        print(out.shape)
        # out = out[[0, 1, 3]]
        # out = out[[0, 1, 2]]
        # out = out[[30, 36, 37]]
        # # out = out[[1, 2, 3]]
        # # np.save(args.pc_dir, out)
        # mask = np.array([1, 2, 0]).repeat([2048, 2048, 2048])
        # out_dict = dict(pred=out[..., :3].reshape(1, -1, 3), pred_seg_mask=out[..., 3].reshape(1, -1))
        # # pickle.dump(out_dict, open(f"{args.pc_dir[:-3]}pkl", 'wb'))
        # visualize_pkl(out_dict, 0)
        # part1 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/shapegf-0.npy")
        # part2 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/shapegf-1.npy")
        # part3 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/shapegf-2.npy")
        
        # out = np.concatenate([part1, part2, part3], axis=1)
        # seg_mask = np.arange(3).repeat([2048, 2048, 2048])[None].repeat(396, axis=0)
        # out_dict = dict(pred=out, pred_seg_mask=seg_mask)
        # visualize_pkl(out_dict, 0)
        # pickle.dump(dict(pred=out, pred_seg_mask=seg_mask), open("/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_ae.pkl", "wb"))
        
        """
        for i in range(out.shape[0]):
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                out[i] = np.matmul(rot_ab, out[i].T).T
            out[i] += (np.random.rand(out.shape[1], 3) - 0.5) * 0.02
        """
        visualize_npy(out[..., :3][FileNotFoundError])
    if args.pc_dir[-3:] == 'pkl':
        out = pickle.load(open(args.pc_dir, "rb"))
        out = {k: v.reshape(-1, *v.shape[-2:]) for k, v in out.items()}
        visualize_pkl(out, args.exp_num, args.params)
        # out = out[[0, 1, 3]]
        # out = out[[0, 1, 2]]
        # out = out[[30, 36, 37]]
        # # out = out[[1, 2, 3]]
        # # np.save(args.pc_dir, out)
        # mask = np.array([1, 2, 0]).repeat([2048, 2048, 2048])
        # out_dict = dict(pred=out[..., :3].reshape(1, -1, 3), pred_seg_mask=out[..., 3].reshape(1, -1))
        # # pickle.dump(out_dict, open(f"{args.pc_dir[:-3]}pkl", 'wb'))
        # visualize_pkl(out_dict, 0)
        # part1 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/shapegf-0.npy")
        # part2 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/shapegf-1.npy")
        # part3 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/shapegf-2.npy")
        
        # out = np.concatenate([part1, part2, part3], axis=1)
        # seg_mask = np.arange(3).repeat([2048, 2048, 2048])[None].repeat(396, axis=0)
        # out_dict = dict(pred=out, pred_seg_mask=seg_mask)
        # visualize_pkl(out_dict, 0)
        # pickle.dump(dict(pred=out, pred_seg_mask=seg_mask), open("/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_ae.pkl", "wb"))
        
        """
        for i in range(out.shape[0]):
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                out[i] = np.matmul(rot_ab, out[i].T).T
            out[i] += (np.random.rand(out.shape[1], 3) - 0.5) * 0.02
        """
        visualize_npy(out[..., :3][FileNotFoundError])
    elif args.pc_dir[-2:] == 'pt':
        import torch
        data = torch.load(args.pc_dir)
        
        pred = data[..., :3]
        seg_mask = data[..., 3]
        for i in range(4):
            mask = seg_mask == i 
            _pred = pred[mask].reshape(1000, 2048, 3)
            np.save(f"/Users/georgenakayama/workspace/visualizers/trained_samples/lion_part_{i}.npy", _pred)
        exit()
        out_dict = dict(pred=out, pred_seg_mask=seg_mask)
        visualize_pkl(out_dict, 0)
        pickle.dump(dict(pred=out, pred_seg_mask=seg_mask), open("/Users/georgenakayama/workspace/visualizers/trained_samples/ae_lion.pkl", "wb"))
        # print(out.shape)
        """
        for i in range(out.shape[0]):
            euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
            rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
            if (np.random.rand(1)[0] > 0.5):
                out[i] = np.matmul(rot_ab, out[i].T).T
            out[i] += (np.random.rand(out.shape[1], 3) - 0.5) * 0.02
        """
        # visualize_npy(out)
        #visualize_npy(out)
    else:
        l = [
            0.0035, 0.0037, 0.0039, 0.0039, 0.0040, 0.0040, 0.0041, 0.0041, 0.0041,
        0.0041, 0.0042, 0.0042, 0.0042, 0.0042, 0.0043, 0.0043, 0.0043, 0.0043,
        0.0043, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044,
        0.0045, 0.0045, 0.0046, 0.0046, 0.0046, 0.0046, 0.0047, 0.0047, 0.0047,
        0.0047, 0.0047, 0.0047, 0.0047, 0.0047, 0.0048, 0.0048, 0.0048, 0.0048,
        0.0048, 0.0048, 0.0048, 0.0048, 0.0048, 0.0048, 0.0048, 0.0048, 0.0048,
        0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0049,
        0.0049, 0.0049, 0.0049, 0.0049, 0.0049, 0.0050, 0.0050, 0.0050, 0.0050,
        0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0051, 0.0051, 0.0051,
        0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051, 0.0051,
        0.0051, 0.0051, 0.0051, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052,
        0.0052, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052,
        0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053,
        0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053, 0.0053,
        0.0053, 0.0054, 0.0054, 0.0054, 0.0054, 0.0054, 0.0054, 0.0054, 0.0054,
        0.0054, 0.0054, 0.0054, 0.0054, 0.0054, 0.0055, 0.0055, 0.0055, 0.0055,
        0.0055, 0.0055, 0.0055, 0.0055, 0.0055, 0.0055, 0.0056, 0.0056, 0.0056,
        0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056, 0.0056,
        0.0056, 0.0056, 0.0056, 0.0056, 0.0057, 0.0057, 0.0057, 0.0057, 0.0057,
        0.0057, 0.0057, 0.0058, 0.0058, 0.0058, 0.0058, 0.0058, 0.0058, 0.0058,
        0.0058, 0.0058, 0.0058, 0.0058, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059,
        0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0060, 0.0060, 0.0060, 0.0060,
        0.0060, 0.0060, 0.0060, 0.0060, 0.0060, 0.0060, 0.0060, 0.0061, 0.0061,
        0.0061, 0.0061, 0.0061, 0.0061, 0.0061, 0.0061, 0.0061, 0.0061, 0.0062,
        0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062, 0.0062,
        0.0062, 0.0063, 0.0063, 0.0063, 0.0063, 0.0063, 0.0063, 0.0063, 0.0063,
        0.0063, 0.0063, 0.0063, 0.0063, 0.0063, 0.0064, 0.0064, 0.0064, 0.0064,
        0.0064, 0.0064, 0.0064, 0.0064, 0.0065, 0.0065, 0.0065, 0.0065, 0.0065,
        0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0066, 0.0067, 0.0067, 0.0067,
        0.0067, 0.0067, 0.0067, 0.0067, 0.0067, 0.0067, 0.0067, 0.0067, 0.0067,
        0.0068, 0.0068, 0.0068, 0.0068, 0.0068, 0.0068, 0.0068, 0.0069, 0.0069,
        0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069, 0.0069,
        0.0070, 0.0070, 0.0070, 0.0070, 0.0070, 0.0070, 0.0070, 0.0070, 0.0070,
        0.0070, 0.0070, 0.0070, 0.0071, 0.0071, 0.0071, 0.0071, 0.0071, 0.0071,
        0.0071, 0.0072, 0.0072, 0.0072, 0.0072, 0.0072, 0.0072, 0.0072, 0.0072,
        0.0072, 0.0072, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073, 0.0073,
        0.0074, 0.0074, 0.0074, 0.0074, 0.0074, 0.0075, 0.0075, 0.0075, 0.0075,
        0.0075, 0.0075, 0.0076, 0.0076, 0.0076, 0.0076, 0.0076, 0.0076, 0.0077,
        0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077, 0.0077,
        0.0077, 0.0077, 0.0077, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078, 0.0078,
        0.0078, 0.0078, 0.0078, 0.0078, 0.0079, 0.0079, 0.0079, 0.0079, 0.0079,
        0.0079, 0.0080, 0.0080, 0.0080, 0.0081, 0.0081, 0.0081, 0.0081, 0.0081,
        0.0081, 0.0082, 0.0082, 0.0082, 0.0082, 0.0082, 0.0082, 0.0082, 0.0083,
        0.0083, 0.0083, 0.0083, 0.0083, 0.0083, 0.0084, 0.0084, 0.0084, 0.0084,
        0.0084, 0.0085, 0.0085, 0.0085, 0.0086, 0.0086, 0.0086, 0.0086, 0.0086,
        0.0086, 0.0086, 0.0086, 0.0086, 0.0087, 0.0087, 0.0087, 0.0088, 0.0088,
        0.0088, 0.0088, 0.0088, 0.0089, 0.0089, 0.0089, 0.0089, 0.0089, 0.0089,
        0.0090, 0.0090, 0.0090, 0.0091, 0.0092, 0.0092, 0.0092, 0.0093, 0.0093,
        0.0093, 0.0095, 0.0095, 0.0095, 0.0096, 0.0096, 0.0096, 0.0096, 0.0096,
        0.0096, 0.0096, 0.0096, 0.0096, 0.0097, 0.0098, 0.0098, 0.0098, 0.0099,
        0.0099, 0.0099, 0.0099, 0.0099, 0.0099, 0.0100, 0.0100, 0.0100, 0.0100,
        0.0101, 0.0101, 0.0102, 0.0102, 0.0102, 0.0102, 0.0103, 0.0103, 0.0105,
        0.0105, 0.0105, 0.0105, 0.0106, 0.0106, 0.0106, 0.0106, 0.0106, 0.0107,
        0.0107, 0.0108, 0.0108, 0.0108, 0.0108, 0.0109, 0.0109, 0.0109, 0.0110,
        0.0110, 0.0111, 0.0112, 0.0112, 0.0112, 0.0112, 0.0112, 0.0113, 0.0115,
        0.0115, 0.0116, 0.0116, 0.0116, 0.0117, 0.0117, 0.0117, 0.0117, 0.0118,
        0.0118, 0.0118, 0.0118, 0.0118, 0.0118, 0.0119, 0.0119, 0.0120, 0.0121,
        0.0122, 0.0123, 0.0124, 0.0124, 0.0125, 0.0126, 0.0127, 0.0127, 0.0128,
        0.0128, 0.0128, 0.0129, 0.0130, 0.0130, 0.0131, 0.0131, 0.0131, 0.0131,
        0.0132, 0.0133, 0.0133, 0.0133, 0.0133, 0.0134, 0.0136, 0.0136, 0.0137,
        0.0138, 0.0139, 0.0140, 0.0140, 0.0140, 0.0142, 0.0142, 0.0143, 0.0143,
        0.0144, 0.0144, 0.0144, 0.0145, 0.0146, 0.0146, 0.0146, 0.0147, 0.0147,
        0.0147, 0.0148, 0.0148, 0.0150, 0.0150, 0.0151, 0.0153, 0.0153, 0.0153,
        0.0154, 0.0154, 0.0154, 0.0156, 0.0158, 0.0158, 0.0160, 0.0161, 0.0162,
        0.0163, 0.0163, 0.0163, 0.0164, 0.0165, 0.0165, 0.0165, 0.0165, 0.0168,
        0.0171, 0.0171, 0.0173, 0.0173, 0.0174, 0.0175, 0.0177, 0.0178, 0.0178,
        0.0178, 0.0180, 0.0183, 0.0183, 0.0183, 0.0184, 0.0184, 0.0184, 0.0185,
        0.0185, 0.0185, 0.0186, 0.0189, 0.0189, 0.0192, 0.0194, 0.0194, 0.0196,
        0.0200, 0.0204, 0.0205, 0.0205, 0.0205, 0.0206, 0.0206, 0.0207, 0.0208,
        0.0208, 0.0209, 0.0213, 0.0215, 0.0217, 0.0219, 0.0219, 0.0220, 0.0220,
        0.0220, 0.0225, 0.0227, 0.0229, 0.0232, 0.0236, 0.0239, 0.0239, 0.0241,
        0.0242, 0.0243, 0.0245, 0.0248, 0.0249, 0.0249, 0.0255, 0.0257, 0.0263,
        0.0264, 0.0268, 0.0272, 0.0275, 0.0278, 0.0288, 0.0296, 0.0302, 0.0306,
        0.0322, 0.0334, 0.0339, 0.0343, 0.0345, 0.0357, 0.0384, 0.0384, 0.0389,
        0.0399, 0.0403, 0.0423, 0.0435, 0.0439, 0.0483, 0.0487, 0.0494, 0.0541,
        0.0567, 0.0583, 0.0599, 0.0604, 0.0631, 0.0758, 0.0868, 0.0953, 0.0988,
        0.1000, 0.1012, 0.1242, 0.1273, 0.1344, 0.1399, 0.1409, 0.1625, 0.1627,
        0.1698, 0.1776, 0.1825, 0.2072, 0.2231, 0.2428, 0.2871, 0.2884, 0.4077,]
        print(sum(l) / len(l))
        # exit()
        out = pickle.load(open(args.pc_dir, 'rb'))
        
        # ref_out_0 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_part/sgf-part0.npy")
        # ref_out_1 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_part/sgf-part1.npy")
        # ref_out_2 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_part/sgf-part2.npy")
        # ref_out_3 = np.load("/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_part/sgf-part3.npy")
        # ids = [9]
        # # ids = [26, 34, 58]
        # n = len(ids)
        # _out = out['pred'][ids]
        # _seg_mask = out['pred_seg_mask'][ids]
        # rest_mask = _seg_mask != 0
        # rest_out = _out[rest_mask].reshape(n, 4096, 3)
        # for i in range(n):
        #     print(ids[i])
        #     # choise = np.random.choice(ref_out_0.shape[0], 100, True)
        #     # chosen_part_0 = ref_out_0[choise]
        #     choise = np.random.choice(ref_out_0.shape[0], 100, True)
        #     chosen_part_0 = ref_out_0[choise]
            
        #     __out = np.concatenate([rest_out[i][None].repeat(100, axis=0), chosen_part_0], axis=1)
        #     print(__out.shape)
        #     np.save(f"/Users/georgenakayama/workspace/visualizers/trained_samples/sgf_part/sample_leg{ids[i]}.npy", __out)
        #     visualize_npy(__out)
        # exit()
        
        # assigned_anchor = out.get('assigned_anchor', None)
        # if 'assigned_anchor' in out.keys():
        #     del out['assigned_anchor']
        # np.save('/Users/georgenakayama/workspace/visualizers/trained_samples/gen_80_sample_10240.pkl', out['pred'])
        # exit()
        # out = {k:rearrange(v, "k h j ... -> (k h j ) ...") for k, v in out.items()}
        # print(out.keys())
        # print(out['ind'])
        # out['pred_seg_mask'] = out['pred_seg_mask'][..., 0]
        # _out = dict()
        # _out['pred'] = out['set1'][..., :3]
        # _out['pred_seg_mask'] = out['set1'][..., 3]
        # _out['input_ref'] = out['set2'][..., :3]
        # _out['ref_seg_mask'] = out['set2'][..., 3]
        # visualize_pkl(dict(pred=out['pred'].reshape(-1, 2048, 3), pred_seg_mask=out['pred_seg_mask'].reshape(-1, 2048)), args.exp_num, args.params)
        # visualize_pkl(out, args.exp_num, args.params)
        _out = dict()
        # print(out['input_ref'].shape, out['ref_seg_mask'].shape)
        # _out['from'] = np.concatenate([out['input_ref'], out['ref_seg_mask'][..., None]], axis=-1)
        # _out['to'] = np.concatenate([out['permuted_ref'], out['permuted_ref_seg_mask'][..., None]], axis=-1)
        # pickle.dump(_out, open("/Users/georgenakayama/workspace/visualizers/trained_samples/interpolation/leg_set.pkl","wb"))
        # visualize_pkl(out, args.exp_num, args.params)
        # visualize_pkl(dict(pred=out[1][..., :3][None], pred_seg_mask=out[1][..., 3][None]), args.exp_num, args.params)
        visualize_pkl(out, args.exp_num, args.params)
        
