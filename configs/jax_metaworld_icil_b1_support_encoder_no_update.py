from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='b_same_family_different_goal_visible_query_goal',
        variant='support_encoder_no_update',
        run_name='B1-support-encoder-no-update-objectcentric',
    )
