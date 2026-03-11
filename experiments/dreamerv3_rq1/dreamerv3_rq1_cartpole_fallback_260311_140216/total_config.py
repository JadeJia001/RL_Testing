exp_config = {
    'env': {
        'manager': {
            'episode_num': float("inf"),
            'max_retry': 1,
            'retry_type': 'reset',
            'auto_reset': True,
            'step_timeout': None,
            'reset_timeout': None,
            'retry_waiting_time': 0.1,
            'cfg_type': 'BaseEnvManagerDict',
            'type': 'base'
        },
        'stop_value': 500,
        'n_evaluator_episode': 1,
        'type': 'cartpole',
        'import_names': ['dizoo.classic_control.cartpole.envs.cartpole_env'],
        'env_id': 'CartPole-v1',
        'collector_env_num': 1,
        'evaluator_env_num': 1
    },
    'policy': {
        'model': {
            'action_shape': 2,
            'actor_dist': 'onehot'
        },
        'learn': {
            'learner': {
                'train_iterations': 1000000000,
                'dataloader': {
                    'num_workers': 0
                },
                'log_policy': True,
                'is_multitask_pipeline': False,
                'only_monitor_rank0': True,
                'hook': {
                    'load_ckpt_before_run': '',
                    'log_show_after_iter': 100,
                    'save_ckpt_after_iter': 10000,
                    'save_ckpt_after_run': True
                },
                'cfg_type': 'BaseLearnerDict'
            },
            'resume_training': False,
            'lambda_': 0.95,
            'grad_clip': 100,
            'learning_rate': 3e-05,
            'batch_size': 8,
            'batch_length': 16,
            'imag_sample': True,
            'slow_value_target': True,
            'slow_target_update': 1,
            'slow_target_fraction': 0.02,
            'discount': 0.997,
            'reward_EMA': True,
            'actor_entropy': 0.0003,
            'actor_state_entropy': 0.0,
            'value_decay': 0.0,
            'update_per_collect': 8
        },
        'collect': {
            'collector': {
                'deepcopy_obs': False,
                'transform_obs': False,
                'collect_print_freq': 100,
                'cfg_type': 'SampleSerialCollectorDict',
                'type': 'sample'
            },
            'n_sample': 1,
            'unroll_len': 1,
            'action_size': 2,
            'collect_dyn_sample': True
        },
        'eval': {
            'evaluator': {
                'eval_freq': 2000,
                'render': {
                    'render_freq': -1,
                    'mode': 'train_iter'
                },
                'figure_path': None,
                'cfg_type': 'InteractionSerialEvaluatorDict',
                'stop_value': 500,
                'n_episode': 1
            }
        },
        'other': {
            'replay_buffer': {
                'type': 'sequence',
                'replay_buffer_size': 100000,
                'deepcopy': False,
                'enable_track_used_data': False,
                'periodic_thruput_seconds': 60,
                'cfg_type': 'SequenceReplayBufferDict'
            },
            'commander': {
                'cfg_type': 'BaseSerialCommanderDict'
            }
        },
        'on_policy': False,
        'cuda': False,
        'multi_gpu': False,
        'bp_update_sync': True,
        'traj_len_inf': False,
        'type': 'dreamer_command',
        'random_collect_size': 200,
        'transition_with_policy_data': False,
        'imag_horizon': 15,
        'cfg_type': 'DREAMERCommandModePolicyDict',
        'import_names': ['ding.policy.mbpolicy.dreamer']
    },
    'world_model': {
        'cfg_type': 'DREAMERWorldModelDict',
        'pretrain': 1,
        'train_freq': 2,
        'model': {
            'state_size': 4,
            'action_size': 2,
            'model_lr': 0.0001,
            'reward_size': 1,
            'hidden_size': 200,
            'batch_size': 8,
            'max_epochs_since_update': 5,
            'dyn_stoch': 32,
            'dyn_deter': 512,
            'dyn_hidden': 512,
            'dyn_input_layers': 1,
            'dyn_output_layers': 1,
            'dyn_rec_depth': 1,
            'dyn_shared': False,
            'dyn_discrete': 32,
            'act': 'SiLU',
            'norm': 'LayerNorm',
            'grad_heads': ['image', 'reward', 'discount'],
            'units': 512,
            'image_dec_layers': 2,
            'reward_layers': 2,
            'discount_layers': 2,
            'value_layers': 2,
            'actor_layers': 2,
            'cnn_depth': 32,
            'encoder_kernels': [4, 4, 4, 4],
            'decoder_kernels': [4, 4, 4, 4],
            'reward_head': 'twohot_symlog',
            'kl_lscale': 0.1,
            'kl_rscale': 0.5,
            'kl_free': 1.0,
            'kl_forward': False,
            'pred_discount': True,
            'dyn_mean_act': 'none',
            'dyn_std_act': 'sigmoid2',
            'dyn_temp_post': True,
            'dyn_min_std': 0.1,
            'dyn_cell': 'gru_layer_norm',
            'unimix_ratio': 0.01,
            'device': 'cpu',
            'obs_type': 'vector',
            'action_type': 'discrete',
            'encoder_hidden_size_list': [128, 64, 64]
        },
        'eval_freq': 250,
        'cuda': False,
        'rollout_length_scheduler': {
            'type': 'linear',
            'rollout_start_step': 20000,
            'rollout_end_step': 150000,
            'rollout_length_min': 1,
            'rollout_length_max': 25
        },
        'type': 'dreamer',
        'import_names': ['ding.world_model.dreamer']
    },
    'exp_name': 'dreamerv3_rq1_cartpole_fallback_260311_140216',
    'seed': 0
}
