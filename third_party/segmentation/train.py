import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import json
from argparse import ArgumentParser



from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from dataset import SegDataset, get_class_weights
from choices import choose_net, get_criterion, get_optimizer, get_lr_scheduler
from predictor import eval_dataset_full, predict_images


def get_train_args():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--net-name", type=str, default='deeplabv3-resnet')
    parser.add_argument("--save-suffix", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out-channels", type=int, default=1)
    parser.add_argument("--erode", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--train-set", type=str, default='/root/autodl-tmp/Segmentation/Data/Mixed/train_Ori')
    parser.add_argument("--val-set", type=str, default='/root/autodl-tmp/Segmentation/Data/Mixed/val')
    parser.add_argument("--test-set", type=str, default='/root/autodl-tmp/Segmentation/Data/Mixed/test')
    parser.add_argument("--test-images", type=str)
    parser.add_argument("--f16", type=str, default=False)
    parser.add_argument("--train-aug", type=str, default=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--opt-name", type=str, default='adamw')
    parser.add_argument("--sch-name", type=str, default='warmup_poly')
    parser.add_argument("--epoch", type=int, default=120)
    parser.add_argument("--save-dir", type=str, default='./results/deeplabv3_resnet')
    parser.add_argument("--pt-dir", type=str)
    parser.add_argument("--pt-stride", type=int, default=40)
    parser.add_argument("--weighting", type=str, default='none')
    parser.add_argument("--eval", type=bool, default=True)
    return parser.parse_args()


def train(args):

    train_set = SegDataset(args.train_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),
                           erode=args.erode, aug=args.train_aug)
    print('Length of train_set:', len(train_set))
    train_dataloader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    train_class_weights = get_class_weights(train_dataloader, out_channels=args.out_channels, weighting=args.weighting)

    if args.eval:
        val_set = SegDataset(args.val_set, num_classes=args.out_channels, appoint_size=(args.height, args.width),
                             erode=0)
        val_dataloader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0
        )
        val_class_weights = get_class_weights(val_dataloader, out_channels=args.out_channels, weighting=args.weighting)
    else:
        val_dataloader, val_class_weights = None, None

    test_set = SegDataset(args.test_set, args.out_channels, appoint_size=(args.height, args.width), erode=0)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    test_class_weights = get_class_weights(test_loader, args.out_channels, args.weighting)

    # Prepare save dir
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.join(
            'results',
            f"{args.save_suffix}-{args.net_name}-h{train_set[0][0].shape[1]}w{train_set[0][0].shape[2]}-"
            f"erode{args.erode}-weighting_{args.weighting}"
        )
    print('Save dir is:{}  Input size is:{}'.format(save_dir, train_set[0][0].shape))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + '/train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Prepare network
    writer = SummaryWriter(save_dir)
    val_dicts = []
    img_size = args.height if args.height else (args.width if args.width else 512)
    net = choose_net(args.net_name, args.out_channels, img_size=img_size).cuda()
    train_criterion = get_criterion(args.out_channels, class_weights=train_class_weights)
    optimizer = get_optimizer(net, args.opt_name)


    steps = len(train_dataloader)
    lr_scheduler = get_lr_scheduler(optimizer, max_iters=args.epoch * steps, sch_name=args.sch_name)

    max_miou = 0
    miou=0

    # Begin to train
    iter_cnt = 0
    for epo in range(args.epoch):
        net.train()
        for batch_id, (batch_data, batch_label) in enumerate(train_dataloader):
            if args.out_channels == 1:
                batch_label = batch_label.float()
            else:
                batch_label = batch_label.squeeze(1)

            iter_cnt += 1
            output = net(batch_data.cuda())
            loss = train_criterion(output, batch_label.cuda())
            iter_loss = loss.item()
            print('Epoch:{} Batch:[{}/{}] Train loss:{}'.format(epo + 1, str(batch_id + 1).zfill(3), steps,
                                                                round(iter_loss, 4)))
            writer.add_scalar('Train loss', iter_loss, iter_cnt)

            optimizer.zero_grad()


            loss.backward()
            optimizer.step()
            if lr_scheduler is not None and args.opt_name != 'adam':
                lr_scheduler.step()

        if args.eval:
            v_loss, (miou, pa) = eval_dataset_full(net.eval(), args.out_channels, val_dataloader,
                                                   class_weights=val_class_weights, save_dir=save_dir, is_val=True)
            writer.add_scalar('Val loss', v_loss, epo + 1)
            writer.add_scalar('Val miou', miou, epo + 1)
            writer.add_scalar('Val pa', pa, epo + 1)
            val_dict_tmp = {}
            val_dict_tmp.setdefault('epoch', epo + 1)
            val_dict_tmp.setdefault('loss', v_loss)
            val_dict_tmp.setdefault('miou', miou)
            val_dict_tmp.setdefault('pa', pa)
            val_dicts.append(val_dict_tmp)

        if miou > max_miou:
            save_file = save_dir + '/' + args.net_name + '_best.pt'
            torch.save(net.state_dict(), save_file)
            print('Saved checkpoint:', save_file)
            max_miou = miou
        if (epo + 1) == args.epoch or (epo + 1) % args.pt_stride == 0 or epo == 0 :
            save_file = save_dir + '/' + args.net_name + '_{}.pt'.format(epo + 1)
            torch.save(net.state_dict(), save_file)
            print('Saved checkpoint:', save_file)

    print(f'the best miou is {max_miou}')
    writer.close()
    with open(save_dir + '/val_log.json', 'w') as f2:
        json.dump(val_dicts, f2, indent=2)

    save_file = save_dir + '/' + args.net_name + '_best.pt'
    net.load_state_dict(torch.load(save_file))
    net.eval()
    with torch.no_grad():
        eval_dataset_full(net, args.out_channels, test_loader, class_weights=test_class_weights, save_dir=save_dir)
        test_images_dir = os.path.join(args.test_set, 'images')
        if not os.path.isdir(test_images_dir):
            test_images_dir = os.path.join(args.test_set, 'image')
        args.test_images = test_images_dir
        predict_images(net, args, dst_size=(512, 512), save_dir=save_dir)


def do_train(args):
    with torch.cuda.device(args.gpu):
        train(args)


def get_choices(args, task):
    sizes = [(512, 512)]
    weightings = ['none']

    if task == 0:
        erodes = [0]
        args.train_set = r'E:\12_guorun\dataset\normal_datasets\seg\aug-disease\train'
        args.val_set = r'E:\12_guorun\dataset\normal_datasets\seg\aug-disease\val'
        args.test_set = r'E:\12_guorun\dataset\normal_datasets\seg\aug-disease\test'

    return args, sizes, erodes, weightings


def search_train(args):
    args.out_channels = 2
    args.epoch = 120
    args.batch_size = 4
    args.gpu = 0
    # train_net_names only support ['CNNSimpleAttention', 'SimpleAttention', 'cpunet', 'munet']
    train_net_names = ['SwinUNetHA']  #   swin_unet   munet   deeplabv3-resnet   transunet  Dy_Seg_Global segformer  CMTFNet  BRAUNet++  SMAFormer  ScaleFormer
    save_suffix = 'aug'
    task = 0
    args, sizes, erodes, weightings = get_choices(args, task=task)

    for net_name in train_net_names:
        for size in sizes:
            for erode in erodes:
                for weighting in weightings:
                    args.weighting = weighting
                    args.erode = erode
                    args.net_name = net_name
                    args.height = int(size[0])
                    args.width = int(size[1])
                    args.save_suffix = save_suffix

                    do_train(args)


if __name__ == "__main__":
    args = get_train_args()
    search_experiment = False

    if search_experiment:
        search_train(args)
    else:
        do_train(args)
