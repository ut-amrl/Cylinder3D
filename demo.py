import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import polar2cat_done

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

class Coda_demo(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="coda.yaml", nusc=None):
        self.return_ref = return_ref
        with open(label_mapping, 'r') as stream:
            codayaml = yaml.safe_load(stream)
        self.learning_map = codayaml['learning_map']
        self.imageset = imageset
        split = codayaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        for i_folder in split:
            print('/'.join([data_path, "3d_raw/os1", str(i_folder)]))
            self.im_idx += absoluteFilePaths('/'.join([data_path, "3d_raw/os1", str(i_folder)]), True)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        index += 1000
        print(self.im_idx[index])
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('raw', 'semantic'),
                                         dtype=np.uint32).reshape((-1, 1))
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)

        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            data_tuple += (raw_data[:, 3],)
        return data_tuple

def build(dataset_config,
          val_dataloader_config,
          grid_size=[480, 360, 32]):
    data_path = train_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]

    SemKITTI = get_pc_model_class(dataset_config['pc_dataset_type'])

    nusc=None
    if "nusc" in dataset_config['pc_dataset_type']:
        from nuscenes import NuScenes
        nusc = NuScenes(version='v1.0-trainval', dataroot=data_path, verbose=True)

    val_pt_dataset = SemKITTI(data_path, imageset=val_imageset,
                              return_ref=val_ref, label_mapping=label_mapping, nusc=nusc)

    val_dataset = Coda_demo(
        val_pt_dataset,
        grid_size=grid_size,
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
    )

    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn_BEV,
                                                     shuffle=val_dataloader_config["shuffle"],
                                                     num_workers=val_dataloader_config["num_workers"])

    return val_dataset_loader


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']

    model_config = configs['model_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = configs['train_params']['model_load_path']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    np.save("demo_results/label_vals", np.array(unique_label_str))

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    val_dataset_loader = build(dataset_config,
                               val_dataloader_config,
                               grid_size=grid_size)

    my_model.eval()
    hist_list = []
    val_loss_list = []
    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                val_dataset_loader):

            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)

            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                    ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()


            # To get points, I get the indices of val_pt_fea[0][:, 3:6] and convert those points to cartesian coordinates (polar2car in adtaset_semantickitti.py)
            # To get labels for the points, I can use predict_labels[0][val_grid[0][:, 0], val_grid[0][:, 1], val_grid[0][:, 2]]
            # The actual labels are at val_pt_labels at the same index
            
            # first get points
            xyz_pol = val_pt_fea[0][:, 3:6]
            xyz = polar2cat_done(xyz_pol)

            # get predicted labels for each point
            predicted = np.array(predict_labels[0][val_grid[0][:, 0], val_grid[0][:, 1], val_grid[0][:, 2]]).reshape((-1, 1))

            # get labels
            actual = np.array(val_pt_labs)[0]

            overall = np.concatenate((xyz, predicted, actual), axis=1)
            overall.dump(f'demo_results/vals_%d' % i_iter_val)

            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())
    iou = per_class_iu(sum(hist_list))
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100
    del val_vox_label, val_grid, val_pt_fea, val_grid_ten

    print('Current val miou is %.3f' %
            (val_miou))
    print('Current val loss is %.3f' %
            (np.mean(val_loss_list)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)

