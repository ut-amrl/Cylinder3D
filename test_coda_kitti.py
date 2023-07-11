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

    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']
    test_dataloader_config = configs['test_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    test_batch_size = test_dataloader_config['batch_size']

    model_config = configs['model_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = "./coda_kitti_test_load/model_load.pt"

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    print("Unique Label:", unique_label)
    print("Unique Label String:", unique_label_str)

    np.save("demo_coda_kitti_test_results/label_vals", np.array(unique_label_str))

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)

    my_model.to(pytorch_device)

    # test_dataset_loader = data_builder.build_test(dataset_config,
    #                                               test_dataloader_config,
    #                                               grid_size=grid_size)

    val_dataset_loader = data_builder.build_val(dataset_config,
                                                val_dataloader_config,
                                                grid_size=grid_size)

    NUM_EXAMPLES = 200
    examples_done = 0

    my_model.eval()
    hist_list = []
    val_loss_list = []

    previous = None

    with torch.no_grad():
        for i_iter_test, (_, test_vox_label, test_grid, test_pt_labs, test_pt_fea) in enumerate(val_dataset_loader):

            test_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                test_pt_fea]
            test_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in test_grid]
            test_label_tensor = test_vox_label.type(torch.LongTensor).to(pytorch_device)

            predict_labels = my_model(test_pt_fea_ten, test_grid_ten, test_batch_size)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()

            # first get points
            xyz_pol = test_pt_fea[0][:, 3:6]
            xyz = polar2cat_done(xyz_pol)

            # get predicted labels for each point
            predicted = np.array(predict_labels[0][test_grid[0][:, 0], test_grid[0][:, 1], test_grid[0][:, 2]]).reshape((-1, 1))

            # get labels
            actual = np.array(test_pt_labs)[0]

            overall = np.concatenate((xyz, predicted, actual), axis=1)
            overall.dump(f'demo_coda_kitti_test_results/vals_%d' % i_iter_test)

            examples_done += 1
            # if (examples_done >= NUM_EXAMPLES):
                # break

            if (previous == None):
                previous = test_grid
    print("----------------------------------")
    print("The entire testing is completed!!!")

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/coda_kitti_test.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)