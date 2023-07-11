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
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data
from dataloader.dataset_semantickitti import polar2cat_done

from utils.load_save_util import load_checkpoint, load_checkpoint_1b1

def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    print("It Get Here")

    configs = load_config_data(config_path)

    print("It Doesn't Get Past")

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = "./model_save_dir/model_save.pt"

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    print("Unique Label:", unique_label)
    print("Unique Label String:", unique_label_str)

    np.save("demo_results/label_vals", np.array(unique_label_str))

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
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

