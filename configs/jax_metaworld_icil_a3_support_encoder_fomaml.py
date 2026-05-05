from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='a_clean_same_instance_hidden_goal',
        variant='support_encoder_fomaml',
        run_name='A3-support-encoder-fomaml-objectcentric',
    )
