from configs.jax_metaworld_icil_object_centric_common import build_config


def get_config():
    return build_config(
        setting='a_oracle_same_instance_query_goal_visible',
        variant='query_only',
        run_name='A5-oracle-query-goal-visible-objectcentric',
    )
