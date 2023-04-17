_base_ = '../imgnet_bs16.py'

# ------- hyper parameters -------

runner_type = 'iter'
steps = 300000
step_drop = 200000

backbone_channels = 2048
hidden_dim = 256
num_patches = 10
aux_loss = True
num_dec_layers = 6

# -------- model --------

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[4],
    frozen_stages=4,
    norm_cfg=dict(type='FrozenBN'))  # add pe in up-detr
neck = dict(
    type='ConditionalTR',
    hidden_dim=hidden_dim,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=num_dec_layers,
    dim_feedforward=2048,
    dropout=0.1,
    normalize_before=False,
    return_intermediate_dec=aux_loss)
position_embedding = dict(type='sine', hidden_dim=hidden_dim)
matcher = dict(
    cost_class=1,
    cost_bbox=5,
    cost_giou=2)
set_criterion = dict(
    losses=['labels', 'boxes', 'cardinality', 'feature'],
    weight_dict=dict(
        loss_ce=1,
        loss_bbox=5,
        loss_giou=2,
        loss_feature=1,
        num_repeat=num_dec_layers),
    eos_coef=0.1)
head = dict(
    type='DETRHead',  # class_embed + bbox_embed + recon
    hidden_dim=hidden_dim,
    feature_recon=True,
    backbone_channels=backbone_channels,
    num_classes=2,
    aux_loss=aux_loss,
    matcher_cfg=matcher,
    criterion_cfg=set_criterion)

model = dict(
    type='UPDETR',  # pe + query
    pretrained='data/model_zoo/resnet/swav_800ep_pretrain_oss.pth.tar',
    freeze_backbone=True,
    query_shuffle=False,
    mask_ratio=0.1,
    num_queries=300,
    num_patches=num_patches,
    backbone_channels=backbone_channels,
    hidden_dim=hidden_dim,
    backbone=backbone,
    position_embedding=position_embedding,
    neck=neck,
    head=head)

# ------- others -------

optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4)
checkpoint_config = dict(interval=5000, by_epoch=False)
lr_config = dict(policy='Step', step=step_drop, gamma=0.1, by_epoch=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
max_iters = steps
