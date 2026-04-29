from configs.eval_query_memory_direct_regression import get_config as get_base_config


def get_config():
    cfg = get_base_config()
    cfg.model.query_encoder_name = 'dp3_query_frame_encoder'
    cfg.model.dp3_query_frame_encoder.use_rgb = True
    cfg.model.dp3_query_frame_encoder.use_mask_id = False
    return cfg
