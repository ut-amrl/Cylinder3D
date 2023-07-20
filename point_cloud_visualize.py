import numpy as np
import open3d as o3d
import time
import os
import cv2
import argparse

from matplotlib import pyplot as plt


def main(args):
    label_path = args.label_path
    input_path = args.input_path
    hide_window = args.hide_window
    video_output_path = args.video_output_path
    no_color = args.no_color
    LABEL_KEYS = np.load(label_path)
    LABEL_COLORS = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [245, 150, 100],
        3: [245, 230, 100],
        4: [250, 80, 100],
        5: [150, 60, 30],
        6: [255, 0, 0],
        7: [180, 30, 80],
        8: [255, 0, 0],
        9: [30, 30, 255],
        10: [200, 40, 255],
        11: [90, 30, 150],
        12: [255, 0, 255],
        13: [255, 150, 255],
        14: [75, 0, 75],
        15: [75, 0, 175],
        16: [0, 200, 255],
        17: [50, 120, 255],
        18: [0, 150, 255],
        19: [170, 255, 150],
        20: [0, 175, 0],
        21: [0, 60, 135],
        22: [80, 240, 150],
        23: [150, 240, 255],
        24: [0, 0, 255],
    }

    color_per_label = {}
    for i in range(len(LABEL_KEYS)):
        color_per_label[i] = list(LABEL_COLORS[i])
        print(color_per_label[i])
        color_per_label[i].reverse()
        print(color_per_label[i])

    for label, val in color_per_label.items():
        if label == 23:
            print("Unknown :", val)
        else:
            print(LABEL_KEYS[label], ":", val)

    def find_colors(labels):
        result = np.zeros((len(labels), 3))
        for i in range(len(labels)):
            curr_color = LABEL_COLORS[labels[i]]
            result[i][0] = curr_color[0]
            result[i][1] = curr_color[1]
            result[i][2] = curr_color[2]
        return result

    def update_point_cloud(file, pcd):
        example = np.load(file, allow_pickle=True)
        point_cloud_vals = example[:, :3]  # point cloud locations
        point_cloud_labels_pred = example[:, 3]  # labels
        if (no_color):
            colors = np.ones((len(point_cloud_labels_pred), 3)) * (70 / 255)
        else:
            colors = find_colors(list(point_cloud_labels_pred)) / 255
        pcd.points = o3d.utility.Vector3dVector(point_cloud_vals)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    def add_point_cloud_to_scene(vis, pcd):
        vis.add_geometry(pcd)

    def update_visualizer_view():
        camera_control = vis.get_view_control()
        camera_control.set_lookat((3.0695789143973324, 1.0532306959878019, 2.9624452627806455))
        camera_control.set_up((0.36022430259164362, 0.011528896318685228, 0.93279447702697993))
        camera_control.set_front((-0.90028904687379541, -0.25764288267730373, 0.35085577818357544))
        camera_control.set_zoom(0.080000000000000002)
        camera_control.change_field_of_view(60)
        render_control = vis.get_render_option()
        render_control.background_color = np.asarray([0, 0, 0])

    total_examples = [file for file in os.listdir(input_path) if file.startswith("vals_")]
    num_examples = len(total_examples)
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=700, width=700, visible=not hide_window)
    start_val = 0
    pcd = o3d.geometry.PointCloud()
    update_point_cloud(input_path + "vals_" + str(start_val), pcd)
    add_point_cloud_to_scene(vis, pcd)
    keep_running = True
    fps_rate = 1e9 / 30
    previous_time_clock = time.time_ns()
    current = 0
    update_visualizer_view()
    vis.update_renderer()
    imgs = []
    fps = 10

    if (video_output_path):
        img = vis.capture_screen_float_buffer(not hide_window)
        # imgs.append(img)
        bounds = img.get_max_bound() - img.get_min_bound()
        width = int(bounds[0])
        height = int(bounds[1])
        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
        img_processed = np.uint8(np.asarray(img) * 255)
        final_img = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
        out.write(final_img)

    while (keep_running):
        if (not hide_window or time.time_ns() - previous_time_clock >= fps_rate):
            # update
            current = ((current + 1) % num_examples)
            if video_output_path and current == 0:
                break
            if current % 30 == 0:
                print("Current ", current)
            update_point_cloud("%s/vals_%d" % (input_path, current + start_val), pcd)
            vis.update_geometry(pcd)
            previous_time_clock = time.time_ns()
            update_visualizer_view()
            vis.update_renderer()
            if (video_output_path):
                img = vis.capture_screen_float_buffer(not hide_window)
                # imgs.append(img)
                # vis.update_renderer()
                img_processed = np.uint8(np.asarray(img) * 255)
                final_img = cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR)
                out.write(final_img)
        keep_running = vis.poll_events()

    vis.destroy_window()
    out.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates a visualization of the inputs and outputs of semantic segmentation for point clouds.')
    parser.add_argument('-v', '--video_output_path', default='')
    parser.add_argument('-w', '--hide_window', action='store_true')
    parser.add_argument('-i', '--input_path', default='demo_results/')
    parser.add_argument('-l', '--label_path', default='demo_results/label_vals.npy')
    parser.add_argument('-nc', '--no_color', action='store_true')
    args = parser.parse_args()

    main(args)
