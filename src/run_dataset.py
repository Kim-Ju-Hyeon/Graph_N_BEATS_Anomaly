import click
import traceback

from easydict import EasyDict as edict
import yaml

from dataset.sea_fog_dataset import Temporal_Graph_Signal


@click.command()
@click.option('--conf_file_path', type=click.STRING, default=None)
def main(conf_file_path):
    config = edict(yaml.load(open(conf_file_path, 'r'), Loader=yaml.FullLoader))

    try:
        loader = Temporal_Graph_Signal()
        loader.preprocess_dataset()
        loader.get_dataset(
            num_timesteps_in=config.forecasting_module.backcast_length,
            num_timesteps_out=config.forecasting_module.forecast_length,
            batch_size=config.train.batch_size)
        print('Dataset Ready')

    except:
        print(traceback.format_exc())


if __name__ == '__main__':
    main()
