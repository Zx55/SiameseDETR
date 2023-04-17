_base_ = '../../base.py'

num_patches = 10

# ------- dataset -------

data_source_cfg = dict(
    type='Coco',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/datasets/mscoco2017/annotations/instances_train2017.json'
data_train_root = 'data/datasets/mscoco2017/train2017/'
dataset_type = 'SiameseDETRDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

base_pipeline = [
    dict(
        type='RandomResizeWithTarget',
        sizes=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        max_size=1333),
]
view_pipeline = [
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
]

crop_pipeline = [
    dict(type='Resize', size=(128, 128)),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0)
        ],
        p=0.5),
]

prefetch = False
if not prefetch:
    view_pipeline.extend([
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg)])
    crop_pipeline.extend([
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg)])

data = dict(
    imgs_per_gpu=4,  # total 16*4=64
    workers_per_gpu=4,
    drop_last=True,
    sampling_replace=False,
    collate_fn=dict(type='SiameseDETRCollateFN'),
    train=dict(
        type=dataset_type,
        data_source=dict(
            ann_file=data_train_list,
            root=data_train_root,
            return_ann=True,
            return_masks=False,
            **data_source_cfg),
        prefetch=False,
        base_pipeline=base_pipeline,
        view_pipeline=view_pipeline,
        anchor_num=num_patches,
        ratio_max=8,
        dice_num=100,
        area_min_ratio=0.05,
        iou=0.5,
        return_crop=True,
        gen_box_method='random',
        views_bbox_stochastic=True,
        crop_pipeline=crop_pipeline))

# ------- others -------

optimizer_config = dict(grad_clip=dict(max_norm=0.1))
