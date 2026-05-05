from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='c_heldout_family_visible_query_goal',
        variant='support_encoder_fomaml',
        run_name='C2-support-encoder-heldout-family-fomaml-objectcentric',
    )
