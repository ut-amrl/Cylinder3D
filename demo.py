import os
import argparse
import sys
import numpy as np
import torch
from pathlib import Path
from strictyaml import Bool, Float, Int, Map, Seq, Str, as_document, load
import os
import numpy as np
from torch.utils import data
import yaml
import pickle
import re
import json
from os.path import join

from builder import data_builder, model_builder, loss_builder


def get_SemKITTI_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    SemKITTI_label_name = dict()
    for i in sorted(list(semkittiyaml['learning_map'].keys()))[::-1]:
        SemKITTI_label_name[semkittiyaml['learning_map'][i]] = semkittiyaml['labels'][i]

    return SemKITTI_label_name


model_params = Map(
    {
        "model_architecture": Str(),
        "output_shape": Seq(Int()),
        "fea_dim": Int(),
        "out_fea_dim": Int(),
        "num_class": Int(),
        "num_input_features": Int(),
        "use_norm": Bool(),
        "init_size": Int(),
    }
)

dataset_params = Map(
    {
        "dataset_type": Str(),
        "pc_dataset_type": Str(),
        "ignore_label": Int(),
        "return_test": Bool(),
        "fixed_volume_space": Bool(),
        "label_mapping": Str(),
        "max_volume_space": Seq(Float()),
        "min_volume_space": Seq(Float()),
    }
)


train_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

val_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

test_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

train_params = Map(
    {
        "model_load_path": Str(),
        "model_save_path": Str(),
        "checkpoint_every_n_steps": Int(),
        "max_num_epochs": Int(),
        "eval_every_n_steps": Int(),
        "learning_rate": Float(),
        "weight_decay": Float(),
        "mixed_fp16": Bool()
    }
)

schema_v4 = Map(
    {
        "format_version": Int(),
        "model_params": model_params,
        "dataset_params": dataset_params,
        "train_data_loader": train_data_loader,
        "val_data_loader": val_data_loader,
        "test_data_loader": test_data_loader,
        "train_params": train_params,
    }
)


SCHEMA_FORMAT_VERSION_TO_SCHEMA = {4: schema_v4}


def load_config_data(path: str) -> dict:
    yaml_string = Path(path).read_text()
    cfg_without_schema = load(yaml_string, schema=None)
    schema_version = int(cfg_without_schema["format_version"])
    if schema_version not in SCHEMA_FORMAT_VERSION_TO_SCHEMA:
        raise Exception(f"Unsupported schema format version: {schema_version}.")

    strict_cfg = load(yaml_string, schema=SCHEMA_FORMAT_VERSION_TO_SCHEMA[schema_version])
    cfg: dict = strict_cfg.data
    return cfg


def polar2cat_done(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])
    y = input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    return np.concatenate((x, y, input_xyz_polar[:, 2].reshape((-1, 1))), axis=1)


def load_checkpoint_1b1(model_load_path, model):
    my_model_dict = model.state_dict()
    pre_weight = torch.load(model_load_path)

    part_load = {}
    match_size = 0
    nomatch_size = 0

    pre_weight_list = [*pre_weight]
    my_model_dict_list = [*my_model_dict]

    for idx in range(len(pre_weight_list)):
        key_ = pre_weight_list[idx]
        key_2 = my_model_dict_list[idx]
        value_ = pre_weight[key_]
        if my_model_dict[key_2].shape == pre_weight[key_].shape:
            # print("loading ", k)
            match_size += 1
            part_load[key_2] = value_
        else:
            print(key_)
            print(key_2)
            nomatch_size += 1

    print("matched parameter sets: {}, and no matched: {}".format(match_size, nomatch_size))

    my_model_dict.update(part_load)
    model.load_state_dict(my_model_dict)

    return model


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    return hist


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
    model_load_path = "./my_things/model_save.pt"
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]
    print("Unique Label:", unique_label)
    print("Unique Label String:", unique_label_str)
    np.save("demo_results/label_vals", np.array(unique_label_str))
    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint_1b1(model_load_path, my_model)
    else:
        print("No Model Found")
        exit(1)
    my_model.to(pytorch_device)
    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True, num_class=num_class, ignore_label=ignore_label)
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)
    my_model.eval()
    hist_list = []
    val_loss_list = []
    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(val_dataset_loader):
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
            predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor, ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
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
                hist_list.append(fast_hist_crop(predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]], val_pt_labs[count], unique_label))
            val_loss_list.append(loss.detach().cpu().numpy())
    iou = per_class_iu(sum(hist_list))
    print('Validation per class iou: ')
    for class_name, class_iou in zip(unique_label_str, iou):
        print('%s : %.2f%%' % (class_name, class_iou * 100))
    val_miou = np.nanmean(iou) * 100
    del val_vox_label, val_grid, val_pt_fea, val_grid_ten
    print('Current val miou is %.3f' % (val_miou))
    print('Current val loss is %.3f' % (np.mean(val_loss_list)))


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='my_things/coda_kitti_test_subset.yaml')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    main(args)
