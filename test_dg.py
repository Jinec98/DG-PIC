import math
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from utils.logger import *
from utils.config import *
from utils.misc import *
import os
from tools import builder
from torch.utils.data import Dataset
import json
from datasets.PairDataset import PairDataset
from models.DGPIC import Encoder
from utils.sampling import knn_point, index_points
from pointnet2_ops import pointnet2_utils
from tqdm import tqdm

def write_plyfile(file_name, point_cloud):
    f = open(file_name + '.ply', 'w')
    init_str = "ply\nformat ascii 1.0\ncomment VCGLIB generated\nelement vertex " + str(len(point_cloud)) + \
               "\nproperty float x\nproperty float y\nproperty float z\n" \
               "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
    f.write(init_str)
    for i in range(len(point_cloud)):
        f.write(str(round(float(point_cloud[i][0]), 6)) + ' ' + str(round(float(point_cloud[i][1]), 6)) + ' ' +
                str(round(float(point_cloud[i][2]), 6)) + '\n')
    f.close()


def sample_farthest_points(points, npoint):
    center_idx = pointnet2_utils.furthest_point_sample(points, npoint).long()
    center_pos = index_points(points, center_idx)
    return center_pos, center_idx

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='whether to set deterministic options for CUDNN backend.')
    # some args
    parser.add_argument('--exp_name', type=str, default='test/DGPIC', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd2', help='loss name')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')

    # dataset
    parser.add_argument('--data_path', type=str, default='data', help='')
    # comment
    parser.add_argument('--comment', type=str, default='default', help='')

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help = 'autoresume training (interrupted by accident)')

    args = parser.parse_args()

    args.experiment_path = args.exp_name

    args.log_name = Path(args.config).stem
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)

    return args

def y_flip(pointcloud1):
    angles = [0, 0, math.pi]
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angles[0]), -np.sin(angles[0])],
                    [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                    [0, 1, 0],
                    [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                    [np.sin(angles[2]), np.cos(angles[2]), 0],
                    [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pointcloud1 = np.dot(pointcloud1.detach().cpu().numpy(), R)
    return pointcloud1

def get_patch(config, pc):
    num_group = config.model.num_group
    group_size = config.model.group_size
    
    pc_center, pc_center_idx = sample_farthest_points(pc, num_group)
    pc_neighborhood_idx = knn_point(group_size, pc, pc_center)
    pc_neighborhood = index_points(pc, pc_neighborhood_idx)
    
    return pc_neighborhood

def find_nearest_sample(test_feature, test_task, feature_dict, simi_rate=0.5):
    B, G, C = test_feature.shape
    sample_dict = {'dataset_idx':[], 'sample_idx':[]}  #[B, ]
    test_feature_mixed = torch.clone(test_feature)
    for b in range(B):
        test_local_feature = test_feature[b]
        test_global_feature = torch.max(test_local_feature, dim=0)[0].unsqueeze(0)
        test_local_norm = F.normalize(test_local_feature)
        test_global_norm = F.normalize(test_global_feature)
        
        nearest_idx = None
        nearest_dataset = None
        nearest_distance = None
        for idx, task in enumerate(feature_dict['task']):
            if task == test_task:
                train_local_prototype = feature_dict['local_prototype'][idx]
                train_global_prototype = feature_dict['global_prototype'][idx]
                train_local_norm = F.normalize(train_local_prototype)
                train_global_norm = F.normalize(train_global_prototype.unsqueeze(0))
                
                local_simi = torch.sum(torch.matmul(test_local_norm, train_local_norm.T)) / (G*G)
                global_simi = torch.matmul(test_global_norm, train_global_norm.T).squeeze()
                
                simi = (1-simi_rate)*local_simi + simi_rate*global_simi
                if nearest_distance == None or nearest_distance < simi:
                    nearest_idx = idx
                    nearest_dataset = feature_dict['dataset'][idx]
                    nearest_distance = simi
        
        # get the nearest domain
        train_local_nearest = feature_dict['local_feature'][nearest_idx].view(-1, G*C)
        train_global_nearest = feature_dict['global_feature'][nearest_idx]
        train_local_nearest_norm = F.normalize(train_local_nearest)
        train_global_nearest_norm = F.normalize(train_global_nearest)

        test_local_norm = test_local_norm.unsqueeze(0).repeat(train_local_nearest.shape[0], 1, 1).view(-1, G*C)
        test_global_norm = test_global_norm.repeat(train_global_nearest.shape[0], 1)
        
        # get the nearset prompt sample from the nearest domain
        local_simi_sample = torch.matmul(test_local_norm, train_local_nearest_norm.T)
        global_simi_sample = torch.matmul(test_global_norm, train_global_nearest_norm.T)
        simi_sample = (1-simi_rate)*local_simi_sample + simi_rate*global_simi_sample
        sample_idx = torch.argmax(simi_sample)
        
        sample_dict['dataset_idx'].append(nearest_idx)
        sample_dict['sample_idx'].append(sample_idx)
        
    return sample_dict          

def eval(args, config, base_model, encoder_model, logger):
    domains = ['modelnet', 'shapenet', 'scannet', 'scanobjectnn']
    source_domains = domains.copy()
    source_domains.remove(config.target_domain)
    tasks = ['reconstruction', 'denoising', 'registration']
    train_domains = {'data':[], 'dataset':[], 'task':[]}
    test_domains = {'data':[], 'dataset':[], 'task':[]}
    
    # load datasets
    for domain in domains:
        if domain == config.target_domain:
            test_dataset_recon = PairDataset(config.dataset.test.others, domain, 'reconstruction', 'test')
            test_dataloader_recon = torch.utils.data.DataLoader(test_dataset_recon, batch_size=config.total_bs * 2,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   num_workers=int(args.num_workers),
                                                   worker_init_fn=worker_init_fn)
            test_domains['data'].append([test_dataset_recon, test_dataloader_recon])
            test_domains['dataset'].append(domain)
            test_domains['task'].append('reconstruction')

            test_dataset_denoi = PairDataset(config.dataset.test.others, domain, 'denoising', 'test')
            test_dataloader_denoi = torch.utils.data.DataLoader(test_dataset_denoi, batch_size=config.total_bs * 2,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   num_workers=int(args.num_workers),
                                                   worker_init_fn=worker_init_fn)
            test_domains['data'].append([test_dataset_denoi, test_dataloader_denoi])
            test_domains['dataset'].append(domain)
            test_domains['task'].append('denoising')
            
            test_dataset_regis = PairDataset(config.dataset.test.others, domain, 'registration', 'test')
            test_dataloader_regis = torch.utils.data.DataLoader(test_dataset_regis, batch_size=config.total_bs * 2,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   num_workers=int(args.num_workers),
                                                   worker_init_fn=worker_init_fn)
            test_domains['data'].append([test_dataset_regis, test_dataloader_regis])
            test_domains['dataset'].append(domain)
            test_domains['task'].append('registration')
        else:
            train_dataset_recon = PairDataset(config.dataset.train.others, domain, 'reconstruction', 'train')
            train_dataloader_recon = torch.utils.data.DataLoader(train_dataset_recon, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
            train_domains['data'].append([train_dataset_recon, train_dataloader_recon])
            train_domains['dataset'].append(domain)
            train_domains['task'].append('reconstruction')

            train_dataset_denoi = PairDataset(config.dataset.train.others, domain, 'denoising', 'train')
            train_dataloader_denoi = torch.utils.data.DataLoader(train_dataset_denoi, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
            train_domains['data'].append([train_dataset_denoi, train_dataloader_denoi])
            train_domains['dataset'].append(domain)
            train_domains['task'].append('denoising')
            
            train_dataset_regis = PairDataset(config.dataset.train.others, domain, 'registration', 'train')
            train_dataloader_regis = torch.utils.data.DataLoader(train_dataset_regis, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
            train_domains['data'].append([train_dataset_regis, train_dataloader_regis])
            train_domains['dataset'].append(domain)
            train_domains['task'].append('registration')


    base_model.eval()
    encoder_model.eval()

    mean_loss = 0
    feature_dict = {'task':[], 'dataset':[], 'local_feature':[], 'global_feature':[], 'local_prototype':[], 'global_prototype':[]}
    load_feature_dict = False

    with torch.no_grad():
        # get train prototype: each dataset and each task
        if not load_feature_dict:
            for idx, [train_dataset, train_dataloader] in enumerate(train_domains['data']):
                train_dataset_name = train_domains['dataset'][idx]
                train_task = train_domains['task'][idx]
                feature_dict['task'].append(train_task)
                feature_dict['dataset'].append(train_dataset_name)
                print('Get prototype from %s of %s task...'%(train_dataset_name, train_task))
                
                global_feature_all = None
                local_feature_all = None
                for b, (pointset, target, rotation, dataset_name) in enumerate(train_dataloader):
                    pointset = pointset.cuda()
                    target = target.cuda()
                    pc_neighborhood= get_patch(config, pointset)
                    
                    # only use source sample
                    local_feature = encoder_model(pc_neighborhood)  #[B, G, C]
                    if local_feature_all == None:
                        local_feature_all = local_feature
                    else:
                        local_feature_all = torch.cat([local_feature_all, local_feature], dim=0)
                global_feature_all = torch.max(local_feature_all, dim=1)[0]
                feature_dict['local_feature'].append(local_feature_all)
                feature_dict['global_feature'].append(global_feature_all)
                feature_dict['local_prototype'].append(torch.mean(local_feature_all, dim=0))
                feature_dict['global_prototype'].append(torch.mean(global_feature_all, dim=0))
            torch.save(feature_dict, os.path.join(args.experiment_path, 'feature_dict.pth'))
            print('Trained feautre dictionaries saved in %s'%(os.path.join(args.experiment_path, 'feature_dict.pth')))
        else:
            feature_dict = torch.load(os.path.join(args.experiment_path, 'feature_dict.pth'))
        

        # test-time DG
        loss_dict = {'loss':[], 'task':[]}
        for idx, [test_dataset, test_dataloader] in enumerate(test_domains['data']):
            test_dataset_name = test_domains['dataset'][idx]
            test_task = test_domains['task'][idx]
            print('Testing in %s task...'%(test_task))
            
            sample_num = 0
            for idx, (pointset, target, rotation, dataset_name) in enumerate(tqdm(test_dataloader)):
                batch_size = pointset.shape[0]
                pointset = pointset.cuda()
                target = target.cuda()
                rotation = rotation.cuda()
                
                # ScanObjectNN need to sample to 1024 points
                pointset, pos_idx = sample_farthest_points(pointset, 1024)
                target = index_points(target, pos_idx)
                
                pc_neighborhood = get_patch(config, pointset)
                
                # find nearest sample in nearest source domain
                local_feature = encoder_model(pc_neighborhood)
                sample_dict = find_nearest_sample(local_feature, test_task, feature_dict)
                
                # get prompt sample
                pointset2 = torch.empty_like(pointset)
                target2 = torch.empty_like(target)
                for b in range(batch_size):
                    train_dataset, train_dataloader = train_domains['data'][sample_dict['dataset_idx'][b]]
                    pointset2[b], target2[b], _, _ = train_dataset[sample_dict['sample_idx'][b]]      
                    if test_task == 'registration':
                        pointset2_origin = torch.clone(target2[b])
                        pointset2_origin = torch.tensor(y_flip(pointset2_origin)).float().cuda()
                        pointset2[b] = torch.matmul(pointset2_origin, rotation[b])
                           
                # eval test sample with test-time dg
                feature_dict_task = {'dataset':[], 'local_feature':[], 'global_feature':[], 'local_prototype':[], 'global_prototype':[]}
                for idx, task in enumerate(feature_dict['task']):
                    if task == test_task:
                        feature_dict_task['dataset'].append(feature_dict['dataset'][idx])
                        feature_dict_task['local_feature'].append(feature_dict['local_feature'][idx])
                        feature_dict_task['global_feature'].append(feature_dict['global_feature'][idx])
                        feature_dict_task['local_prototype'].append(feature_dict['local_prototype'][idx])
                        feature_dict_task['global_prototype'].append(feature_dict['global_prototype'][idx])
                        
                _, rebuild_points, loss = base_model(pointset2, pointset, target2, target, test=True, feature_dict=feature_dict_task, mix_rate=0.9)
                
                rebuild_points, _ = sample_farthest_points(rebuild_points, target.shape[1])
                loss = base_model.module.loss_func(rebuild_points, target)
                mean_loss += loss.item()*1000 * batch_size
                sample_num += batch_size
            mean_loss /= sample_num
            
            print_log('[TEST %s] loss = %.4f'%(test_task, mean_loss), logger=logger)
            loss_dict['loss'].append(mean_loss)
            loss_dict['task'].append(test_task)

    return loss_dict

def main():
    # args
    args = get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}-{args.seed}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # config
    config = get_config(args, logger = logger)
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)

    print_log(args.comment)

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)  # seed + rank, for augmentation

    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)

    encoder_model = Encoder(encoder_channel = config.model.transformer_config.encoder_dims)
    
    if args.use_gpu:
        device = torch.device('cuda')
        base_model.to(device)
        encoder_model.to(device)
        
    print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()
    encoder_model = nn.DataParallel(encoder_model).cuda()

    # load encoder model
    base_model_keys = []
    encoder_model_keys = []
    for key in base_model.state_dict():
        if key.split('.')[0] == 'MAE_encoder' and key.split('.')[1] == 'encoder':
            base_model_keys.append(key)
    for key in encoder_model.state_dict():
        encoder_model_keys.append(key)
    for (base_model_key, encoder_model_key) in zip(base_model_keys, encoder_model_keys):
        encoder_model.state_dict()[encoder_model_key].copy_(base_model.state_dict()[base_model_key])

    loss_dict = eval(args, config, base_model, encoder_model, logger)
    print('All tasks are done!')
    for idx in range(len(loss_dict['task'])):
        print('Test in %s, loss = %.4f'%(loss_dict['task'][idx], loss_dict['loss'][idx]))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
