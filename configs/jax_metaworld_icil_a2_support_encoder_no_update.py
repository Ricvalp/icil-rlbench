from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='a_clean_same_instance_hidden_goal',
        variant='support_encoder_no_update',
        run_name='A2-support-encoder-no-update-objectcentric',
    )
