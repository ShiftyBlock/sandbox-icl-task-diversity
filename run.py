import logging
from absl import app, flags
from ml_collections import config_flags

import icl.utils as u
from icl.train import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")
flags.mark_flag_as_required("config")


def main(_):
    logging.info("Starting training")

    config = u.filter_config(FLAGS.config)
    train(config)


if __name__ == "__main__":
    app.run(main)
