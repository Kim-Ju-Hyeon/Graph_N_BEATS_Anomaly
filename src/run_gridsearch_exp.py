import click
from runner.runner import Runner
import traceback
from utils.logger import setup_logging
import os
from utils.train_helper import set_seed, mkdir, edict2dict
import datetime

import pytz
from easydict import EasyDict as edict
import yaml
import time


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    exp_list = ['classification', 'regression_all', 'regression_vis']
    bool_list = [True, False]
    for loss_type in exp_list:
        for combine_loss in bool_list:
            if loss_type == 'classification' and combine_loss:
                pass
            elif loss_type == 'classification' and not combine_loss:
                pass
            else:
                for _ in range(1):
                    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))
                    config.train.loss_type = loss_type
                    config.train.combine_loss = combine_loss

                    now = datetime.datetime.now(pytz.timezone('Asia/Seoul'))
                    sub_dir = now.strftime('%m%d_%H%M%S')
                    config.seed = set_seed(config.seed)

                    if combine_loss:
                        config.exp_name = str(config.exp_name) + '_' + loss_type + '_combine_loss'
                    else:
                        config.exp_name = str(config.exp_name) + '_' + loss_type + '_single_loss'

                    config.exp_dir = os.path.join(config.exp_dir, config.exp_name)
                    config.exp_sub_dir = os.path.join(config.exp_dir, sub_dir)
                    config.model_save = os.path.join(config.exp_sub_dir, "model_save")

                    mkdir(config.model_save)

                    save_name = os.path.join(config.exp_sub_dir, 'config.yaml')
                    yaml.dump(edict2dict(config), open(save_name, 'w'), default_flow_style=False)

                    log_file = os.path.join(config.exp_sub_dir, "log_exp_{}.txt".format(config.seed))
                    logger = setup_logging('INFO', log_file, logger_name=str(config.seed))
                    logger.info("Writing log file to {}".format(log_file))
                    logger.info("Exp instance id = {}".format(config.exp_name))

                    try:
                        runner = Runner(config=config)
                        runner.train()
                        runner.test()

                    except:
                        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
