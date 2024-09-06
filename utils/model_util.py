from model.mrm import MRM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps
from utils.parser_util import get_cond_mode


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data):
    model = MRM(**get_model_args_mrm(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args_mrm(args):

    # default args
    # clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = 'none'
    # if hasattr(data.dataset, 'num_actions'):
    #     num_actions = data.dataset.num_actions
    # else:
    num_actions = 1

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 28
    nfeats = 1

    # if args.dataset == 'humanml':
    #     data_rep = 'hml_vec'
    #     njoints = 263
    #     nfeats = 1
    # elif args.dataset == 'kit':
    #     data_rep = 'hml_vec'
    #     njoints = 251
    #     nfeats = 1
    args.use_latent = args.human_data_path is not None

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            # 'use_fp16': args.use_fp16, 'use_latent': args.use_latent,
            # 'sigma_max': args.sigma_max, 'sigma_min': args.sigma_min,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec}#, 'dataset': args.dataset}

def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )
