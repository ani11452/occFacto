# model settings
model = dict(
    type='AnchorDiffAE',
    encoder=dict(
        type='PartEncoderForTransformerDecoder',
        encoder=dict(
            type='PointNetV2',
            zdim=256,
            point_dim=3,
            per_part_mlp=True,
        ),
        n_class=4,
        kl_weight=5e-4,
        fit_loss_type=4,
        fit_loss_weight=1.0,
        use_flow=True,
        latent_flow_depth=14, 
        latent_flow_hidden_dim=256,
        include_z=False,
        include_part_code=True,
        include_params=True,
        use_gt_params=True,
        kl_weight_annealing=False,
        min_kl_weight=1e-7,
        kl_weight_annealing_end_epoch=4000,
        gen=True,
        prior_var=1.0
    ),
    diffusion=dict(
        type='AnchoredDiffusion',
        net = dict(
            type='TransformerNet',
            in_channels=3,
            out_channels=3,
            n_heads=8,
            d_head=16,
            depth=5,
            dropout=0.2,
            context_dim=256 + 6,
            n_class=4,
            class_cond=True,
            use_linear=True,
            cat_params_to_x=True,
            use_checkpoint=False,
            single_attn=True,
            cat_class_to_x=True,
        ),
        beta_1=1e-4,
        beta_T=.02,
        k=1.0,
        res=False,
        mode='linear',
        use_beta=False,
        rescale_timesteps=False,
        model_mean_type="epsilon",
        learn_variance=True,
        loss_type='mse',
        include_anchors=False,
        
        classifier_weight=1.,
        guidance=False,
        ddim_sampling=False,
        ddim_nsteps=25,
        ddim_discretize='quad',
        ddim_eta=1.
    ),
    sampler = dict(type='Uniform'),
    num_anchors=4,
    num_timesteps=100,
    npoints = 2048,
    
    ret_traj = False,
    ret_interval = 10,
    forward_sample=False,
    drift_anchors=False,
    interpolate=False,
    save_weights=False,
    save_dir="/mnt/disk3/wang/diffusion/anchorDIff/work_dirs/pn_aware_attn_both_feat_guidance_200T_no_augment/checkpoints"
)

dataset = dict(
    train=dict(
        type="ShapeNetSegPart",
        batch_size = 128,
        split='trainval',
        root='/orion/u/w4756677/datasets/diffFacto_data',
        npoints=2048,
        scale_mode='shape_unit',
        part_scale_mode='shape_canonical',
        eval_mode='ae',
        clip=False,
        num_workers=4,
        class_choice='Chair',
    ),
    val=dict(
        type="ShapeNetSegPart",
        batch_size= 64,
        split='test',
        root='/orion/u/w4756677/datasets/diffFacto_data',
        npoints=2048,
        shuffle=False,
        scale_mode='shape_unit',
        part_scale_mode='shape_canonical',
        clip=False,
        eval_mode='ae',
        num_workers=0,
        class_choice='Chair',
        save_only=True
    ),
)

optimizer = dict(type='Adam', lr=0.002, weight_decay=0.)

scheduler = dict(
    type='LinearLR',
    start_lr=2e-3,
    end_lr = 1e-4,
    start_epoch=4000,
    end_epoch=8000,
)


logger = dict(
    type="RunLogger")

# when we the trained model from cshuan, image is rgb
save_num_batch = 1
max_epoch = 8000
eval_interval = 500
checkpoint_interval = 500
log_interval = 50
max_norm=10
# eval_both=True