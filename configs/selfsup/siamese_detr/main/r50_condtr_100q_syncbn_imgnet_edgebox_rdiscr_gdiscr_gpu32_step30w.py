_base_ = '../imgnet_edgebox_crop_bs8.py'

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

transformer = dict(
    type='ConditionalTR',
    hidden_dim=hidden_dim,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=num_dec_layers,
    dim_feedforward=2048,
    dropout=0.1,
    normalize_before=False,
    return_intermediate_dec=aux_loss)

pred_head = dict(
    type='SiameseDETRPredictHead',
    aux_loss=aux_loss,
    hidden_dim=hidden_dim,
    size_average=True,
    feature_recon=True,
    backbone_channels=backbone_channels,
    matcher_cfg=dict(
        cost_class=1,
        cost_bbox=5,
        cost_giou=2))

encoder_head = dict(
    type='SiameseDETREncoderGlobalLatentHead',
    projector=dict(
        type='NonLinearNeckV3',
        in_channels=256,
        hid_channels=256,
        out_channels=256,
        sync_bn=True,
        with_avg_pool=True),
    predictor=dict(
        type='NonLinearNeckV2',
        in_channels=256,
        hid_channels=256,
        out_channels=256,
        sync_bn=True,
        with_avg_pool=False),
    size_average=True)

position_embedding = dict(type='sine', hidden_dim=hidden_dim)

model = dict(
    type='SiameseDETR',  # pe + query
    pretrained='data/model_zoo/resnet/swav_800ep_pretrain_oss.pth.tar',
    freeze_backbone=True,
    query_shuffle=False,
    num_queries=100,
    num_patches=num_patches,
    box_disturbance=0.1,
    backbone_channels=backbone_channels,
    feature_recon=True,
    hidden_dim=hidden_dim,
    weight_dict=dict(
        loss_enc_global_contra=10,
        loss_ce=1,
        loss_bbox=5,
        loss_giou=2,
        loss_feature=5,
        num_repeat=num_dec_layers),
    backbone=backbone,
    position_embedding=position_embedding,
    transformer=transformer,
    pred_head=pred_head,
    encoder_head=encoder_head)

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
