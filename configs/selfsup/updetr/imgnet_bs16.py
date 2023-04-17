_base_ = '../../base.py'

# ------- hyper parameters -------

num_patches = 10

# ------- dataset -------

data_source_cfg = dict(
    type='ImageNet',
    memcached=True,
    mclient_path='/mnt/lustre/share/memcached_client')
data_train_list = 'data/datasets/imagenet/meta/train.txt'
data_train_root = 'data/datasets/imagenet/train'
dataset_type = 'UPDETRDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

image_pipeline = [
    dict(
        type='RandomResizeWithTarget',
        sizes=[320, 336, 352, 368, 400, 416, 432, 448, 464, 480],
        max_size=600),
]
patch_pipeline = [
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
        p=0.5)
]

prefetch = False
if not prefetch:
    image_pipeline.extend([
        dict(type='ToTensorWithTarget'),
        dict(type='NormalizeWithTarget', **img_norm_cfg)])
    patch_pipeline.extend([
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg)])
        
data = dict(
    imgs_per_gpu=16,  # total 16*16=256
    workers_per_gpu=4,
    drop_last=True,
    sampling_replace=False,
    collate_fn=dict(type='UPDETRCollateFN'),
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            return_label=False,
            **data_source_cfg),
        image_pipeline=image_pipeline,
        patch_pipeline=patch_pipeline,
        num_patches=num_patches,
        prefetch=prefetch))

# ------- others -------

optimizer_config = dict(grad_clip=dict(max_norm=0.1))
