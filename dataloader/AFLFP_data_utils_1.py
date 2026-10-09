import torch


def set_up_datasets(args):
    if args.dataset == 'AFLFP':
        import dataloader.AFLFP_1 as Dataset
    args.Dataset = Dataset
    return args


def get_dataloader(args):
    flip_p = float(getattr(args, "flip_p", 0.5))

    trainset = args.Dataset.AFLFP(
        root=args.data_path,
        split_txt_root=args.data_split_path,
        train=True,
        cache_labels=True,
        flip_p=flip_p,
    )
    testset = args.Dataset.AFLFP(
        root=args.data_path,
        split_txt_root=args.data_split_path,
        train=False,
        cache_labels=True,
        flip_p=0.0,
    )

    print('trainset size: ', len(trainset))
    print('testset size: ', len(testset))

    num_workers = int(getattr(args, "workers", 0))

    train_kwargs = dict(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_kwargs = dict(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if num_workers > 0:
        train_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))
        test_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    trainloader = torch.utils.data.DataLoader(**train_kwargs)
    testloader = torch.utils.data.DataLoader(**test_kwargs)

    return trainset, testset, trainloader, testloader


def get_labelname(args):
    if args.dataset == 'AFLFP':
        label_nms = ["Left_AU02", "Left_AU04", "Left_AU06", "Left_AU15", "Left_AU43",
                     "Right_AU02", "Right_AU04", "Right_AU06", "Right_AU15", "Right_AU43"]
    print(args.dataset, 'dataset, label_nms: ', label_nms)
    args.label_nms = label_nms
    return args