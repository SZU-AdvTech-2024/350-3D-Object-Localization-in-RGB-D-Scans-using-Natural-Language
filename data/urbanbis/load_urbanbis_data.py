""" 
Modified from: https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py

Load Scannet scenes with vertices and ground truth labels for semantic and instance segmentations
"""

# python imports
import math
import os, sys, argparse
import inspect
import json
import pdb
import numpy as np
import open3d as o3d

def export(main_path, city_name, area_number, output_file):
    """ points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    save_lines = {}
    the_last_lines = []
    the_last_points = []
    the_last_colours = []
    the_last_semantic_label = -1
    the_last_instance_label = -1
    lines = []
    with open(f"/{main_path}/{city_name}/{area_number}.txt", "r") as file:
        for line in file:
            x, y, z, r, g, b, _, _, _ = map(float, line.split())
            semantic_label, instance_label, building_label = map(int, line.split()[-3:])
    
            if semantic_label in [0, 1, 2, 3, 4, 5, -100]:
                semantic_label = 0
                instance_label = 0
            else:
                if building_label not in [-100, 7]:
                    semantic_label += (building_label + 1)
    
            if the_last_semantic_label == -1 and the_last_instance_label == -1:
                the_last_semantic_label = semantic_label
                the_last_instance_label = instance_label
    
            if semantic_label != the_last_semantic_label or instance_label != the_last_instance_label:
                if the_last_lines:  # 保存之前的实例
                    save_lines[the_last_instance_label] = {
                        "semantic_label": the_last_semantic_label,
                        "instance_label": the_last_instance_label,
                        "lines": the_last_lines,
                        "points": the_last_points,
                        "colours": the_last_colours,
                    }
                # 更新实例
                the_last_semantic_label = semantic_label
                the_last_instance_label = instance_label
                the_last_lines = []
                the_last_points = []
                the_last_colours = []
    
            # 添加当前点
            the_last_points.append([x, y, z])
            the_last_colours.append([r, g, b])
            the_last_lines.append([x, y, z, r, g, b, semantic_label, instance_label])

            lines.append([x, y, z, r, g, b, semantic_label, instance_label, building_label])
    
        # 保存最后一个实例
        if the_last_lines:
            save_lines[the_last_instance_label] = {
                "semantic_label": the_last_semantic_label,
                "instance_label": the_last_instance_label,
                "lines": the_last_lines,
                "points": the_last_points,
                "colours": the_last_colours,
            }
    print(len(lines))      
    mesh_vertices = np.array([sublist[:6] for sublist in lines], dtype=np.float32)
    label_ids = np.array([sublist[6] for sublist in lines], dtype=np.int32)
    instance_ids = np.array([sublist[7] for sublist in lines], dtype=np.int32)
    aligned_instance_bboxes = np.zeros((len(save_lines.keys()),8))
    #print(list(save_lines.keys()))
    print(aligned_instance_bboxes.shape)
    instance_id = 0
    for keys_i, values_i in save_lines.items():
        if values_i["semantic_label"] in [0, 1, 2, 3, 4, 5]:
            continue
        category_pcd_i = o3d.geometry.PointCloud()
        category_pcd_i.points = o3d.utility.Vector3dVector(np.array(values_i["points"]))
        try:
            aabb = category_pcd_i.get_axis_aligned_bounding_box()
            aabb_center = aabb.get_center()
            aabb_extent = aabb.get_extent()
            bbox = [

                aabb_center[0], aabb_center[1], aabb_center[2], 
                aabb_extent[0], aabb_extent[1], aabb_extent[2], 
                values_i["semantic_label"], values_i["instance_label"]]
            aligned_instance_bboxes[instance_id,:] = bbox
            instance_id += 1
        except Exception as e:
            print(e)
            continue
        
    aligned_instance_bboxes = aligned_instance_bboxes[~np.all(aligned_instance_bboxes == 0, axis=1)]
    print(aligned_instance_bboxes.shape)
    print(aligned_instance_bboxes)
    if output_file is not None:
        np.save(output_file+'_vert.npy', mesh_vertices)
        np.save(output_file+'_sem_label.npy', label_ids)
        np.save(output_file+'_ins_label.npy', instance_ids)
        np.save(output_file+'_aligned_bbox.npy', aligned_instance_bboxes)

    return mesh_vertices, label_ids, instance_ids, aligned_instance_bboxes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_path', required=True, help='path to area')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--area_name', required=True, help='the name of area')
    opt = parser.parse_args()

    city_name = opt.area_name.split("_")[0]
    area_number = "_".join(opt.area_name.split("_")[1:])

    export(opt.main_path, city_name, area_number, opt.output_file)

if __name__ == '__main__':
    main()
