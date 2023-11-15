from functools import partial 
import open3d as o3d
import numpy as np 
from utils.general import *
import os
from colour import Color
from gif_maker import make_gif
import time

def normalize_point_clouds(pc):
    shift = pc.mean(axis=0).reshape(1, 3)
    scale = pc.flatten().std().reshape(1, 1)
    return (pc - shift) / scale


def visualize_pkl(out, exp_num, interpolate_params=False, assigned_anchor=None):
    T_list = list(out.keys())
    len_T = len(T_list)
    print(T_list)
    T_id = 0
    id = 0
    # print(out['set1'].shape)
    c = Color("blue")
    colors = np.array(list(map(lambda c: list(c.rgb), c.range_to(Color("red"), steps=4)))).reshape(4, 3)
    # npoints = out[T_list[T_id]].shape[1]
    _colors = np.array([[1.,1.,1.]])
    # colors[3] = _colors
    pred_colors = colors[:,None].repeat(2048 // 4, axis=1).reshape(2048, 3)
    n = out[T_list[T_id]].shape[0]
    print(n)
    pcd = o3d.geometry.PointCloud()
    color_id = (T_id, id)
    translate=False
    fix_color=False
    use_mask=False
    add_text=True
    voxel = False
    pc = out[T_list[T_id]][id]
    pcd.points = o3d.utility.Vector3dVector(pc)
    
    pcd.colors = o3d.utility.Vector3dVector(normalize(pc))
    text_pcd = text_3d(T_list[T_id], [0, 1, 0], direction=[0, 0, 1], degree=270, font_size=30)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(visible=True) #works for me with False, on some systems needs to be true
    vis.add_geometry(pcd)
    vis.add_geometry(text_pcd)
    view_contrl = vis.get_view_control()
    view_contrl.rotate(*view_dict['top'])
    view_params = view_contrl.convert_to_pinhole_camera_parameters()
    ren = vis.get_render_option()
    if T_list[T_id] == 'anchors':
        ren.point_size = 5.
    ren.background_color=np.array([1.,1.,1.])
    ren.point_color_option = o3d.visualization.PointColorOption.Color
    ren.show_coordinate_frame = True
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    
    
    
    def incrementT(_vis):
        nonlocal T_id, color_id
        T_id = ( 1 + T_id) % len_T
        if 'seg_mask' in str(T_list[T_id]):
            incrementT(_vis)
        if not fix_color:
            color_id = (T_id, id)
        update_vis(_vis)
    
    def decrementT(_vis):
        nonlocal T_id, color_id
        T_id = (T_id - 1) % len_T
        if 'seg_mask' in str(T_list[T_id]):
            decrementT(_vis)
        if not fix_color:
            color_id = (T_id, id)
        update_vis(_vis)
        
    def toggleText(_vis):
        nonlocal add_text
        add_text = not add_text
        update_vis(_vis)

    def incrementid(_vis):
        nonlocal id, color_id
        id  += 1
        id %= n
        if not fix_color:
            color_id = (T_id, id)
        update_vis(_vis)
    
    def decrementid(_vis):
        nonlocal id, color_id
        id -= 1
        id %= n
        if not fix_color:
            color_id = (T_id, id)
        update_vis(_vis)
        
    def save_view(_vis):
        nonlocal view_params
        view_contrl = _vis.get_view_control()
        view_params = view_contrl.convert_to_pinhole_camera_parameters()
    
    def update_vis(_vis):
        save_view(_vis)
        print(f'Changed to model id = {id}, T = {T_list[T_id]}, color_id: ({color_id[0], color_id[1]}), fix_color set to {fix_color}')
        _vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        pc = out[T_list[T_id]][id]
        pc = pc.clip(-5, 5)
        if translate and T_list[T_id] == 'pred anchor drift sample 0':
            
            anchors = out.get("anchor sample 0", None)[id]
            if use_mask:
                mask = out.get('seg_mask', None)
                anchors = anchors[mask[id]]
            else:
                anchors = anchors[:, None].repeat(2048 // 4, axis=1).reshape(2048, 3)
            pc = pc - anchors
        pcd.points = o3d.utility.Vector3dVector(pc)
        if 'pred' in str(T_list[T_id]):
            print(out[T_list[T_id]][id].max(0) - out[T_list[T_id]][id].min(0))
        if fix_color:
            if T_list[T_id] == 'anchors':
                pcd.colors = o3d.utility.Vector3dVector(colors)
            elif 'input' in str(T_list[T_id]):
                mask_key = T_list[T_id].replace('input', 'seg_mask')
                mask = out.get(mask_key, None)
                print(colors.shape)
                if mask is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors[mask[id]])
            else:
                if use_mask:
                    print(colors.shape)
                    mask = out.get('pred_seg_mask', None).astype(int)
                        
                    print(mask.shape)
                    if mask is not None:
                        pcd.colors = o3d.utility.Vector3dVector(colors[mask[id]])
                else:
                    pcd.colors = o3d.utility.Vector3dVector(pred_colors)
        else:
            pcd.colors = o3d.utility.Vector3dVector(normalize(pc))

        if voxel:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.5)
        text_pcd = text_3d(T_list[T_id], [0, 1, 0], direction=[0, 0, 1], degree=270, font_size=30)
        vis.add_geometry(voxel_grid if voxel else pcd)
        if add_text:
            vis.add_geometry(text_pcd)
        view_contrl = _vis.get_view_control()
        view_contrl.convert_from_pinhole_camera_parameters(view_params)
        ren = _vis.get_render_option()
        ren.light_on = True
        if T_list[T_id] == 'anchors':
            ren.point_size = 10
        else:
            ren.point_size = 5.
        ren.background_color=np.array([1.,1.,1.])
        ren.point_color_option = o3d.visualization.PointColorOption.Color
        ren.show_coordinate_frame = False
        _vis.update_geometry(pcd)
        _vis.poll_events()
        _vis.update_renderer()
        return False

    def setColor(_vis):
        nonlocal color_id, fix_color 
        color_id = (T_id, id)
        fix_color = not fix_color
        print(f'color set to T = {T_list[T_id]} id = {id}, fix_color set to {fix_color}')
        update_vis(_vis)
        
    def toggleTranslate(_vis):
        nonlocal translate
        translate = not translate 
        update_vis(_vis)
    
    def useSegmaskAsColor(_vis):
        nonlocal color_id, fix_color, use_mask
        use_mask = not use_mask
        update_vis(_vis)

    def voxelize(_vis):
        nonlocal voxel

        voxel = not voxel 
        update_vis(_vis)
        
    def getSnapshots(_vis):
        nonlocal id
        save_dir = f"/Users/georgenakayama/workspace/visualizers/snapshots"
        gif_save_dir = f"/Users/georgenakayama/workspace/visualizers/gifs/exp{exp_num}"
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(gif_save_dir, exist_ok=True)
        
        for j in range(n):
            for i in range(10):
                path = os.path.join(save_dir, f'T{i}.png')
                capture_screen_shot(_vis, path)
                incrementT(_vis)
            for i in range(10):
                decrementT(_vis)
            save_path = f"{gif_save_dir}/interpolate_exp{exp_num}_sample{j}" if not interpolate_params else f"{gif_save_dir}/interpolate_params_exp{exp_num}_sample{j}"
            make_gif(save_dir, save_path)
            incrementid(_vis)
            
   
    vis.register_key_callback(ord("S"), partial(capture_screen_shot))
    vis.register_key_callback(ord("D"), partial(incrementT))
    vis.register_key_callback(ord("A"), partial(decrementT))
    vis.register_key_callback(ord("E"), partial(incrementid))
    vis.register_key_callback(ord("Q"), partial(decrementid))
    vis.register_key_callback(ord("H"), partial(getSnapshots))
    vis.register_key_callback(ord("C"), partial(setColor))
    vis.register_key_callback(ord("V"), partial(voxelize))
    vis.register_key_callback(ord("T"), partial(toggleText))
    vis.register_key_callback(ord("M"), partial(useSegmaskAsColor))
    vis.register_key_callback(ord("Z"), partial(toggleTranslate))
    vis.run()
    vis.destroy_window()

