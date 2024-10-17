import sys
sys.path.append('.')

from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from datasets.PairDataset import PairDataset
import torch.utils.data
from utils.misc import *
import math

class Loss_Metric:
    def __init__(self, loss = 0 ):
        if type(loss).__name__ == 'dict':
            self.loss = loss['loss']
        else:
            self.loss = loss

    def better_than(self, other):
        if self.loss < other.loss:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['loss'] = self.loss
        return _dict


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
    pointcloud1 = np.dot(pointcloud1, R)
    return pointcloud1

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset

    domains = ['modelnet', 'shapenet', 'scannet', 'scanobjectnn']
    source_domains = domains.copy()
    source_domains.remove(config.target_domain)
    tasks = ['reconstruction', 'denoising', 'registration']
    train_domains = {'data':[], 'dataset':[], 'task':[]}
    test_domains = {'data':[], 'dataset':[], 'task':[]}
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

    
    print('Merging source domains...')
    recon_source_dataset = None
    denoi_source_dataset = None
    regis_source_dataset = None
 
    
    for idx, [train_dataset, train_dataloader] in enumerate(train_domains['data']):
        train_dataset_name = train_domains['dataset'][idx]
        train_task = train_domains['task'][idx]
        if train_task == 'reconstruction':
            if recon_source_dataset == None:
                recon_source_dataset = train_dataset
            else:
                recon_source_dataset = torch.utils.data.ConcatDataset([recon_source_dataset, train_dataset])
        elif train_task == 'denoising':
            if denoi_source_dataset == None:
                denoi_source_dataset = train_dataset
            else:
                denoi_source_dataset = torch.utils.data.ConcatDataset([denoi_source_dataset, train_dataset])
        elif train_task == 'registration':
            if regis_source_dataset == None:
                regis_source_dataset = train_dataset
            else:
                regis_source_dataset = torch.utils.data.ConcatDataset([regis_source_dataset, train_dataset])
    
    recon_source_dataloader = torch.utils.data.DataLoader(recon_source_dataset, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
    denoi_source_dataloader = torch.utils.data.DataLoader(denoi_source_dataset, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
    regis_source_dataloader = torch.utils.data.DataLoader(regis_source_dataset, batch_size=config.total_bs,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=int(args.num_workers),
                                             worker_init_fn=worker_init_fn)
    sources_dataloaders = [{'dataloader': recon_source_dataloader, 'task': 'reconstruction'}, 
                           {'dataloader': denoi_source_dataloader, 'task': 'denoising'}, 
                           {'dataloader': regis_source_dataloader, 'task': 'registration'}]

    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # parameter setting
    start_epoch = 0
    best_metrics = Loss_Metric(100000.)
    metrics = Loss_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Loss_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0
        find_other_datasets =True  # True for ours, False for PIC
        copy_target = False  # True for baseline
        
        base_model.train()  # set model to training mode
        
        for train_dataloader_dict in sources_dataloaders:
            train_dataloader = train_dataloader_dict['dataloader']
            task = train_dataloader_dict['task']
            print('Training model in %s task...'%(task))

            n_batches = len(train_dataloader)
            for idx, (pointset1_pc, target1, rotation1, dataset_name) in enumerate(train_dataloader):
                num_iter += 1
                n_itr = epoch * n_batches + idx
                if copy_target:
                    pointset2_pc = target1.clone()
                    target2 = target1.clone()
                else:
                    # random create pair
                    datasets_choice = [source_domains.copy()] * config.total_bs
                    pointset2_pc = torch.empty_like(pointset1_pc)
                    target2 = torch.empty_like(target1)
                    rotation2 = torch.empty_like(rotation1)
                    for b in range(config.total_bs):
                        if find_other_datasets:
                            datasets_choice_b = datasets_choice[b].copy()
                            datasets_choice_b.remove(dataset_name[b])
                        else:
                            datasets_choice_b = [dataset_name[b]]
                        dataset2_name = random.choice(datasets_choice_b)
                        # get the prompt sample
                        for i in range(len(train_domains['task'])):
                            if train_domains['task'][i] == task and train_domains['dataset'][i] == dataset2_name:
                                train_dataset2 = train_domains['data'][i][0]
                                pointset2_idx = random.randint(0, len(train_dataset2)-1)
                                pointset2_pc[b], target2[b], rotation2[b], _ = train_dataset2[pointset2_idx]
                                if task == 'registration':
                                    pointset2_origin = torch.clone(target2[b])
                                    pointset2_origin = torch.from_numpy(y_flip(pointset2_origin)).float()
                                    pointset2_pc[b] = torch.matmul(pointset2_origin, rotation1[b])  # align rotation with pointset1
                    
                data_time.update(time.time() - batch_start_time)

                pointset1_pc = pointset1_pc.cuda()
                pointset2_pc = pointset2_pc.cuda()
                
                target1 = target1.cuda()
                target2 = target2.cuda()
                            
                _, _, loss = base_model(pointset1_pc, pointset2_pc, target1, target2)
                try:
                    loss.backward()
                except:
                    loss = loss.mean()
                    loss.backward()

                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                losses.update([loss.item()*1000])

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 20 == 0:
                    print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                                (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                                ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
            optimizer.param_groups[0]['lr']), logger = logger)


        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()