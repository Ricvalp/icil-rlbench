import os

from ml_collections import ConfigDict


def get_config():
    cfg = ConfigDict()

    # Number of optimizer steps to trace before stopping.
    cfg.trace_n_steps = 20

    # Perfetto/Chrome-trace JSON output.
    cfg.output_dir = os.environ.get(
        "ICIL_PROFILE_TRACE_DIR",
        os.path.join("output_data_playground_v3", ".profiles"),
    )
    cfg.trace_filename = os.environ.get(
        "ICIL_PRETRAIN_PROFILE_TRACE_FILE",
        "pretrain_trace.json",
    )

    # Profiler options.
    cfg.trace_cuda = True
    cfg.record_shapes = True
    cfg.profile_memory = True
    cfg.with_stack = True
    cfg.with_flops = False

    return cfg
