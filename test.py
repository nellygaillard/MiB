import utils
import os
from utils.logger import Logger
import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed
from dataset import VOCSegmentationIncremental
from dataset import transform
from metrics import StreamSegMetrics
from segmentation_module import make_model
import task

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    train_transform = transform.Compose([
        transform.RandomResizedCrop(opts.crop_size, (0.5, 2.0)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = transform.Compose([
            transform.Resize(size=opts.crop_size),
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    else:
        # no crop, batch size = 1
        val_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    labels, labels_old, path_base = tasks.get_task_labels(opts.dataset, opts.task, opts.step)
    labels_cum = labels_old + labels

    if opts.dataset == 'voc':
        dataset = VOCSegmentationIncremental
    else:
        raise NotImplementedError

    if opts.overlap:
        path_base += "-ov"

    if not os.path.exists(path_base):
        os.makedirs(path_base, exist_ok=True)

    test_dst = dataset(root=opts.data_root, train=opts.val_on_trainset, transform=val_transform,
                       labels=list(labels_cum),
                       idxs_path=path_base + f"/test_on_{image_set}-{opts.step}.npy")

    return  test_dst, len(labels_cum)



def main(opts):

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_size', type=int, default=320)
    parser.add_argument('--dataset', type=str, default='voc', help='Name of the dataset')
    parser.add_argument('--task', type=str, default='15-5s', help="Task to be executed (default: 19-1)")
    parser.add_argument("--logdir", type=str, default='./logs',
                        help="path to Log directory (default: ./logs)")
    parser.add_argument("--name", type=str, default='Experiment',
                        help="name of the experiment - to append to log directory (default: Experiment)")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="verbose option")
    parser.add_argument("--step", type=int, default=0,
                        help="The incremental step in execution (default: 0)")
    parser.add_argument("--visualize", action='store_false', default=True,
                        help="visualization on tensorboard (def: Yes)")
    parser.add_argument("--sample_num", type=int, default=0,
                        help='number of samples for visualization (default: 0)')
    parser.add_argument('--ckpt_path', type=str, default=None, required=True, help='The path to the pretrained weights of model')
    parser.add_argument('--data', type=str, default='path/to/data', help='Path of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--backbone', type=str, default='resnet50', help='The backbone model you are using')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--num_classes', type=int, default=21, help='num of object classes (with void)')
    parser.add_argument('--loss', type=str, default='bce', help='loss function, default crossentropy')
    parser.add_argument("--random_seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate (default: 0.005)")

    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)

    os.makedirs("/content/drive/MyDrive/MLDL/CKPT/checkpoints/step", exist_ok=True)

    rank = 0
    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    # if rank == 0:
    logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # xxx Set up dataloader
    test_dst, n_classes = get_dataset(opts)

    # reset the seed, this revert changes in random seed
    random.seed(opts.random_seed)

    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")


    # dobbiamo creare il modello relativo al file pth da cui vogliamo leggerlo, quindi ci serve sapere lo step
    # a cui si riferisce (per istanziare il corretto numero di classifiers), il dataset e il tipo di task (numero di nuove classi)
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))


    # non ci serve invece instanziare il vecchio modello, perché non siamo in fase di training

    logger.debug(model)

    # xxx Set up optimizer
    #params = []
    #params.append({"params": filter(lambda p: p.requires_grad, model.parameters()),
    #               'weight_decay': opts.weight_decay})

    optimizer = torch.optim.SGD(params, lr=opts.lr, momentum=0.9, nesterov=True)

    #if opts.lr_policy == 'poly':
    #    scheduler = utils.PolyLR(optimizer, max_iters=opts.epochs * len(train_loader), power=opts.lr_power)
    #elif opts.lr_policy == 'step':
    #    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    #else:
    #    raise NotImplementedError
    logger.debug("Optimizer:\n%s" % optimizer)

    val_metrics = StreamSegMetrics(n_classes)
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))

    logger.info("*** Test the model on all seen classes...")
    # make data loader
    test_loader = data.DataLoader(test_dst,
                                  shuffle=False,
                                  batch_size=opts.batch_size if opts.crop_val else 1,
                                  num_workers=opts.num_workers)

    # load model
    model = make_model(opts, classes=tasks.get_per_task_classes(opts.dataset, opts.task, opts.step))
    ckpt = f"/content/drive/MyDrive/MLDL/CKPT/checkpoints/step/{task_name}_{opts.name}_{opts.step}.pth"
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    logger.info(f"*** Model restored from {ckpt}")
    del checkpoint
    trainer = Trainer(model, None, opts=opts)

    model.eval()

    sample_ids = np.random.choice(len(test_loader), opts.sample_num, replace=False)  # sample idxs for visualization

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    val_loss, val_score, ret_samples = trainer.validate(loader=test_loader,
                                                        metrics=val_metrics,
                                                        ret_samples_ids=sample_ids,
                                                        logger=logger)
    logger.print("Done test")
    logger.info(f"*** End of Test, Total Loss={val_loss[0] + val_loss[1]},"
                f" Class Loss={val_loss[0]}, Reg Loss={val_loss[1]}")
    logger.info(val_metrics.to_str(val_score))
    logger.add_table("Test_Class_IoU", val_score['Class IoU'])
    logger.add_table("Test_Class_Acc", val_score['Class Acc'])
    logger.add_figure("Test_Confusion_Matrix", val_score['Confusion Matrix'])
    results["T-IoU"] = val_score['Class IoU']
    results["T-Acc"] = val_score['Class Acc']
    logger.add_results(results)

    for k, (img, target, lbl) in enumerate(ret_samples):
        img = (denorm(img) * 255).astype(np.uint8)
        target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
        lbl = label2color(lbl).transpose(2, 0, 1).astype(np.uint8)

    concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
    logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

    logger.add_scalar("T_Overall_Acc", val_score['Overall Acc'], opts.step)
    logger.add_scalar("T_MeanIoU", val_score['Mean IoU'], opts.step)
    logger.add_scalar("T_MeanAcc", val_score['Mean Acc'], opts.step)

    logger.close()


if __name__ == '__main__':
    params = [
        '--ckpt_path', 'path/to/ckpt',
        '--data', 'path/to/data'
        '--cuda', '0',
        '--backbone', 'resnet50',
        '--num_classes', '21',
        '--sample_num', '3'
    ]
    main(params)
