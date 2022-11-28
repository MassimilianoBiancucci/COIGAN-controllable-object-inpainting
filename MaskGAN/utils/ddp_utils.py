import os, sys
import functools

import logging

LOGGER = logging.getLogger(__name__)

def get_has_ddp_rank():
    master_port = os.environ.get('MASTER_PORT', None)
    node_rank = os.environ.get('NODE_RANK', None)
    local_rank = os.environ.get('LOCAL_RANK', None)
    world_size = os.environ.get('WORLD_SIZE', None)
    has_rank = master_port is not None or node_rank is not None or local_rank is not None or world_size is not None
    return has_rank


def handle_ddp_subprocess():
    def main_decorator(main_func):
        @functools.wraps(main_func)
        def new_main(*args, **kwargs):
            # Trainer sets MASTER_PORT, NODE_RANK, LOCAL_RANK, WORLD_SIZE
            parent_cwd = os.environ.get('TRAINING_PARENT_WORK_DIR', None)
            has_parent = parent_cwd is not None
            has_rank = get_has_ddp_rank()
            assert has_parent == has_rank, f'Inconsistent state: has_parent={has_parent}, has_rank={has_rank}'

            if has_parent:
                # we are in the worker
                sys.argv.extend([
                    f'hydra.run.dir={parent_cwd}',
                    # 'hydra/hydra_logging=disabled',
                    # 'hydra/job_logging=disabled'
                ])
            # do nothing if this is a top-level process
            # TRAINING_PARENT_WORK_DIR is set in handle_ddp_parent_process after hydra initialization

            main_func(*args, **kwargs)
        return new_main
    return main_decorator


def handle_ddp_parent_process():
    parent_cwd = os.environ.get('TRAINING_PARENT_WORK_DIR', None)
    has_parent = parent_cwd is not None
    has_rank = get_has_ddp_rank()
    assert has_parent == has_rank, f'Inconsistent state: has_parent={has_parent}, has_rank={has_rank}'

    if parent_cwd is None:
        os.environ['TRAINING_PARENT_WORK_DIR'] = os.getcwd()

    return has_parent