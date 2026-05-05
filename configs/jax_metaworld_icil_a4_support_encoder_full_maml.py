from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='a_clean_same_instance_hidden_goal',
        variant='support_encoder_full_maml',
        run_name='A4-support-encoder-full-maml-objectcentric',
    )
