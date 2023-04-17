_base_ = '../imgnet_edgebox_crop_bs4.py'

# ------- hyper parameters -------

runner_type = 'iter'
steps = 300000
step_drop = 200000

multi_scale_features_backbone_strides = (8, 16, 32)
multi_scale_features_backbone_num_channels = (512, 1024, 2048)
backbone_channels = sum(multi_scale_features_backbone_num_channels)
num_queries = 300

hidden_dim = 256
num_patches = 10
aux_loss = True
num_dec_layers = 6

# -------- model --------

backbone = dict(
    type='ResNet',
    depth=50,
    in_channels=3,
    out_indices=[2, 3, 4],
    frozen_stages=4,
    norm_cfg=dict(type='FrozenBN'))  # add pe in up-detr

transformer = dict(
    type='DeformableTR',
    d_model=hidden_dim,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=num_dec_layers,
    dim_feedforward=1024,
    dropout=0.1,
    activation="relu",
    return_intermediate_dec=True,
    num_feature_levels=4,
    dec_n_points=4,
    enc_n_points=4,
    two_stage=False,
    two_stage_num_proposals=num_queries,
    num_queries=num_queries)

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
        in_channels=256*len(multi_scale_features_backbone_strides),
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
    num_queries=300,
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
        loss_feature=3,
        num_repeat=num_dec_layers),
    backbone=backbone,
    position_embedding=position_embedding,
    transformer=transformer,
    pred_head=pred_head,
    encoder_head=encoder_head,
    multi_scale_features=True,
    multi_scale_features_backbone_strides=multi_scale_features_backbone_strides,
    multi_scale_features_backbone_num_channels=multi_scale_features_backbone_num_channels,
)

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
