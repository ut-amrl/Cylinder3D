import argparse
import glob
from pathlib import Path
import time
import copy
import json
import os
import yaml

ROS_DEBUG_FLAG = True

import numpy as np

# ROS IMPORTS
import rospy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
import std_msgs.msg

from queue import Queue

import torch
from builder import model_builder
from config.config import load_config_data

from dataloader.dataset_semantickitti import cart2polar, polar2cat, polar2cat_done, nb_process_label

from utils.load_save_util import load_remapped_checkpoint

pc_msg_queue = Queue()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--point_cloud_topic', type=str, default='/coda/ouster/points',
                        help='specify the point cloud ros topic name')
    parser.add_argument('--output_topic', type=str, default='/coda/cylinder3d/points')
    parser.add_argument('-y', '--config_path', default='config/coda.yaml')

    args = parser.parse_args()

    return args

def point_cloud_callback(msg):
    if ROS_DEBUG_FLAG:
        pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        pc_list = list(pc_data)
        pc_np = np.array(pc_list, dtype=np.float32)

        print("Received point cloud with shape ", pc_np.shape)

    pc_msg_queue.put(msg)

def create_model(configs, pytorch_device):

    model_config = configs['model_params']
    model_load_path = configs['train_params']['model_load_path']

    my_model = model_builder.build(model_config)

    if os.path.exists(model_load_path):
        my_model = load_remapped_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)

    my_model.eval()

    return my_model

def create_dataset_format(configs):
    dataset_config = configs['dataset_params']
    model_config = configs['model_params']

    grid_size = model_config['output_shape']

    dataset_format = cylinder_dataset(
        grid_size=grid_size,
        flip_aug=True,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,
        scale_aug=True,
        transform_aug=True
    )

    return dataset_format

def run_model(model, pc_msg, dataset_format, pytorch_device):
    pc_data = point_cloud2.read_points(pc_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    pc_list = list(pc_data)
    pc_np = np.array(pc_list, dtype=np.float32) 

    random_labels = np.expand_dims(np.zeros_like(pc_np[:, 0], dtype=int), axis=1)

    with torch.no_grad():
        
        _, test_vox_label, test_grid, test_pt_labs, test_pt_fea = dataset_format.generate_cylinder_input((pc_np[:, :3], random_labels, pc_np[:, 3]))

        test_pt_fea_ten = [torch.from_numpy(test_pt_fea).type(torch.FloatTensor).to(pytorch_device)]
        test_grid_ten = [torch.from_numpy(test_grid).to(pytorch_device)]

        predict_labels = model(test_pt_fea_ten, test_grid_ten, 1)
        predict_labels = torch.argmax(predict_labels, dim=1)
        predict_labels = predict_labels.cpu().detach().numpy()

        predicted = np.array(predict_labels[0][test_grid[:, 0], test_grid[:, 1], test_grid[:, 2]]).reshape((-1, 1)) # (131072, 1) list with class for each point

    return pc_np, predicted

def pub_pc_to_rviz(pc, pc_pub, ts, point_type="x y z", frame_id="os_sensor", publish=True):
    if not isinstance(ts, rospy.Time):
        ts = rospy.Time.from_sec(ts)

    def add_field(curr_bytes_np, next_pc, field_name, fields):
        """
        curr_bytes_np - expect Nxbytes array
        next_pc - expects Nx1 array
        datatype - uint32, uint16
        """
        field2dtype = {
            "x":    np.array([], dtype=np.float32),
            "y":    np.array([], dtype=np.float32),
            "z":    np.array([], dtype=np.float32),
            "i":    np.array([], dtype=np.float32),
            "t":    np.array([], dtype=np.uint32),
            "re":   np.array([], dtype=np.uint16),
            "ri":   np.array([], dtype=np.uint16),
            "am":   np.array([], dtype=np.uint16),
            "ra":   np.array([], dtype=np.uint32),
            "r":    np.array([], dtype=np.int32),
            "g":    np.array([], dtype=np.int32),
            "b":    np.array([], dtype=np.int32)
        }
        field2pftype = {
            "x": PointField.FLOAT32,  "y": PointField.FLOAT32,  "z": PointField.FLOAT32,
            "i": PointField.FLOAT32,  "t": PointField.UINT32,  "re": PointField.UINT16,  
            "ri": PointField.UINT16, "am": PointField.UINT16, "ra": PointField.UINT32,
            "r": PointField.INT32, "g": PointField.INT32, "b": PointField.INT32
        }
        field2pfname = {
            "x": "x", "y": "y", "z": "z", 
            "i": "intensity", "t": "t", 
            "re": "reflectivity",
            "ri": "ring",
            "am": "ambient", 
            "ra": "range",
            "r": "r",
            "g": "g",
            "b": "b"
        }

        #1 Populate byte data
        dtypetemp = field2dtype[field_name]

        next_entry_count = next_pc.shape[-1]
        next_bytes = next_pc.astype(dtypetemp.dtype).tobytes()

        next_bytes_width = dtypetemp.itemsize * next_entry_count
        next_bytes_np = np.frombuffer(next_bytes, dtype=np.uint8).reshape(-1, next_bytes_width)

        all_bytes_np = np.hstack((curr_bytes_np, next_bytes_np))

        #2 Populate fields
        pfname  = field2pfname[field_name]
        pftype  = field2pftype[field_name]
        pfpos   = curr_bytes_np.shape[-1]
        fields.append(PointField(pfname, pfpos, pftype, 1))
        
        return all_bytes_np, fields

    #1 Populate pc2 fields
    pc = pc.reshape(-1, pc.shape[-1]) # Reshape pc to N x pc_fields
    all_bytes_np = np.empty((pc.shape[0], 0), dtype=np.uint8)
    all_fields_list = []
    field_names = point_type.split(" ")
    for field_idx, field_name in enumerate(field_names):
        next_field_col_np = pc[:, field_idx].reshape(-1, 1)
        # import pdb; pdb.set_trace()

        all_bytes_np, all_fields_list = add_field(
            all_bytes_np, next_field_col_np, field_name, all_fields_list
        )

    #2 Make pc2 object
    pc_msg = PointCloud2()
    pc_msg.width        = 1
    pc_msg.height       = pc.shape[0]

    pc_msg.header            = std_msgs.msg.Header()
    pc_msg.header.stamp      = ts
    pc_msg.header.frame_id   = frame_id

    pc_msg.point_step = all_bytes_np.shape[-1]
    pc_msg.row_step     = pc_msg.width * pc_msg.point_step
    pc_msg.fields       = all_fields_list
    pc_msg.data         = all_bytes_np.tobytes()
    pc_msg.is_dense     = True

    if publish:
        pc_pub.publish(pc_msg)

    return pc_msg

def publish_data(pc_pub, input_data, predicted_data, label_mapping):
    with open(label_mapping, 'r') as stream:
        codayaml = yaml.safe_load(stream)

    inverse_learning_map = codayaml['learning_map_inv']
    color_map = codayaml['color_map']

    output_data = []

    for i in range(len(predicted_data)):
        # get current color
        predicted_value = predicted_data[i][0]
        original_value = inverse_learning_map[predicted_value]
        curr_color_arr = color_map[original_value]

        output_data.append(curr_color_arr)

    output_data = np.array(output_data)

    data = np.hstack((input_data, output_data))

    pub_pc_to_rviz(data, pc_pub, rospy.Time.now(), point_type='x y z i r g b')
        
def main():
    args = parse_config()

    pytorch_device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    config_path = args.config_path
    configs = load_config_data(config_path)

    #1 Load model
    cylinder3d_model = create_model(configs, pytorch_device)
    dataset_format = create_dataset_format(configs)

    #2 Initialize ROS
    pc_topic = args.point_cloud_topic
    rospy.init_node('CODaROSDetector', anonymous=True)
    rospy.Subscriber(pc_topic, PointCloud2, point_cloud_callback)
    semseg_cyl3d = rospy.Publisher('/coda/semseg_cyl3d', MarkerArray, queue_size=10)

    output_topic = args.output_topic
    pc_pub = rospy.Publisher(output_topic, PointCloud2, queue_size=5)

    while not rospy.is_shutdown():
        if not pc_msg_queue.empty():
            pc_msg = pc_msg_queue.get()
            
            input_data, predicted_data = run_model(cylinder3d_model, pc_msg, dataset_format, pytorch_device)
            # input_data = np.zeros((25, 4))
            # predicted_data = np.array(list(range(0, 25))).reshape(-1, 1)
            publish_data(pc_pub, input_data, predicted_data, configs['dataset_params']['label_mapping'])
            
            # publish ros data with color for each label
            # create a new method
            if ROS_DEBUG_FLAG:
                print("Working on outputting data for next msg")


class cylinder_dataset:
    def __init__(self, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4):
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def generate_cylinder_input(self, data):
        'Generates one sample of data'
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        data_tuple += (grid_ind, labels, return_fea)

        return data_tuple
        
if __name__ == '__main__':
    main()
