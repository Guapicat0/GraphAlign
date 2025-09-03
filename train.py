import os
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets import model_selector

from utils.utils_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory,EvalHistory
from utils.utils import show_config
from utils.utils_fit import fit_one_epoch,fit_one_epoch_scene_graph
from torch.utils.data import SequentialSampler
# path confirm
current_file_path = os.getcwd()
parent_directory = os.path.dirname(current_file_path)


if __name__ == "__main__":

    Cuda = True
    distributed = True
    sync_bn = False
    fp16 = False
    model_name = "OVSGTR_ALIGN"
    pretrained = True
    model_path = "model_data/ViT-B-32.pt"

    # AGIQA3K ,AIGCIQA2023
    dataset_name = 'SG-AIGCIQA2023'
    train_dataset_path = f"train.json"
    val_dataset_path   = f"val.json"
    input_shape = [224, 224]
    # 512，512
    Init_Epoch = 0
    UnFreeze_Epoch = 100
    batch_size = 128
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    save_period = 10
    save_dir = f'logs/{dataset_name}/D3/noise'
    eval_flag = True
    eval_period = 2
    loss = "MSE"
    num_workers = 8
    ngpus_per_node = torch.cuda.device_count()

    train_dataset_path = os.path.join(parent_directory,train_dataset_path)
    val_dataset_path = os.path.join(parent_directory, val_dataset_path)

    # select dataloader for model
    if model_name in model_selector.graph_models:
        print("Detect Graph-IQA model，use Graph DataLoader...")
        from utils.graph_dataloader import *
    elif model_name in model_selector.iqa_models:
        print("Detect vlm model，use IQA DataLoader...")
        from utils.dataloader import *
    else:
        raise ValueError(f"Not find '{model_name}'，please check！")

    if distributed:
        dist.init_process_group(backend="nccl", init_method='env://')

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)

        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:

        device = torch.device('cuda:0' )
        local_rank = 0

    #model = model_selector.get_model(model_name,ckpt=model_path,device=device).train()
    model = model_selector.get_model(model_name, device=device).train()

    if model_name in model_selector.graph_models:
        print("Detect Graph-IQA model，use Graph training")
    elif model_name in model_selector.iqa_models:
        print("Detect vlm model，use IQA training...")

    if not pretrained:
        weights_init(model)

    # ----------------------#
    #   loss
    # ----------------------#
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
        eval_history = EvalHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        eval_history = None


    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:

            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model = model.cuda(0)  #
            model_train = torch.nn.DataParallel(model, device_ids=[0])  # GPU 0
            cudnn.benchmark = True

    # ---------------------------#
    #   load dataset txt
    # ---------------------------#
    AGIQA_Dataset = get_dataset_class(dataset_name)
    train_dataset = AGIQA_Dataset(json_files=train_dataset_path, input_shape=input_shape)
    val_dataset   = AGIQA_Dataset(json_files=val_dataset_path, input_shape=input_shape)
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    if local_rank == 0:
        show_config(
            net=model, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            batch_size=batch_size, Init_lr=Init_lr, Min_lr=Min_lr,
            optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir,
            num_workers=num_workers, num_train=num_train, num_val=num_val
        )


    nbs = 16
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-3 if optimizer_type == 'adam' else 5e-3
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # ---------------------------------------#
    #   optimizer_type
    # ---------------------------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay,eps=1e-3),
        'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay),
        'adamw': optim.AdamW(model.parameters(),Init_lr_fit,betas=(momentum, 0.999),weight_decay=weight_decay,eps=1e-3)
    }[optimizer_type]

    # ---------------------------------------#
    #   lr
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)


    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("dataset is too small")



    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True,
                     drop_last=False, collate_fn=dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=False, collate_fn=dataset_collate,sampler=SequentialSampler(val_sampler))


        # ---------------------------------------#
        #   train
        # ---------------------------------------#
    for epoch in range(Init_Epoch, UnFreeze_Epoch):

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        # different_model has different_training type
        if model_name in model_selector.graph_models:
            fit_one_epoch_scene_graph(model_train, model, loss_history, eval_history,optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, loss,
                          fp16, scaler, save_period, save_dir, local_rank)
        elif model_name in model_selector.iqa_models:
            fit_one_epoch(model_train, model, loss_history, eval_history, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, loss,
                          fp16, scaler, save_period, save_dir, local_rank)
        else:
            raise ValueError(f"not find '{model_name}', not training, please check")


        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
        eval_history.writer.close()

