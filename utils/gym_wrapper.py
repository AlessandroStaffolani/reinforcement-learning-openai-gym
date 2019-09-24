from gym import wrappers, logger
import os

from definitions import ROOT_DIR


def wrap_env(env, task_name, logger_level=logger.INFO):
    logger.set_level(logger_level)

    outdir = os.path.join(ROOT_DIR, 'logs/' + task_name + '-results')

    return wrappers.Monitor(env, directory=outdir, force=True)
