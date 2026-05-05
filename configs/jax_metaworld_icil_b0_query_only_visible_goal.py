from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='b_same_family_different_goal_visible_query_goal',
        variant='query_only',
        run_name='B0-query-only-visible-goal-objectcentric',
    )
