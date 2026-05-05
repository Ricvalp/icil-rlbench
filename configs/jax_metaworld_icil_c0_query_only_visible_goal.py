from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='c_heldout_family_visible_query_goal',
        variant='query_only',
        run_name='C0-query-only-heldout-family-visible-goal-objectcentric',
    )
