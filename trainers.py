import os

from absl import flags
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
import datetime

from datasets import datasets_dict, BaseDataGenerator
from models import Model


flags.DEFINE_bool("restore", False, "Whether to restore previous params from checkpoint.", short_name="r")
flags.DEFINE_bool("predict", False, "Whether to only do predictions.", short_name="p")
flags.DEFINE_string("restore_date", "", "Which date folder to restore checkpoint from. If empty, will get newest")
flags.DEFINE_string("restore_time", "", "Which time folder to restore checkpoint from. If empty, will get newest")
flags.DEFINE_integer("restore_epoch", 1, "Which epoch to restore.")

flags.DEFINE_integer("gpus", 1, "Number of GPUs to use",
                     lower_bound=0)
flags.DEFINE_integer("eval_frequency", 10, "How often to evaluate the model.")

FLAGS = flags.FLAGS


class ModelTrainer:
    def __init__(self):
        # Folders
        parent_folder = "results/"

        now = datetime.datetime.now()

        if not(FLAGS.restore or FLAGS.predict):
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M")
            parent_folder += f"{date}/{time}/"
        else:
            date = FLAGS.restore_date if FLAGS.restore_date != "" else sorted(os.listdir(parent_folder))[-1]
            parent_folder += f"{date}/"
            if not os.path.exists(parent_folder):
                raise RuntimeError(f"{parent_folder} does not exist")

            time = FLAGS.restore_time if FLAGS.restore_time != "" else sorted(os.listdir(parent_folder))[-1]
            parent_folder += f"{time}/"

            if not os.path.exists(parent_folder):
                raise RuntimeError(f"{parent_folder} does not exist")

        parent_folder += f"data_{FLAGS.dataset}/"
        parent_folder += f"prior_{FLAGS.prior}/"

        # if FLAGS.same_diffusion:
        #     parent_folder += f"drift_{FLAGS.drift}/diffusion_{FLAGS.diffusion}_same/"
        # else:
        #     parent_folder += f"drift_{FLAGS.drift}/diffusion_{FLAGS.diffusion}/"
        parent_folder += f"solver={FLAGS.solver}_numSteps={FLAGS.num_steps}/"

        self.summary_folder = parent_folder + "summary/"
        self.results_folder = parent_folder + "results/"
        self.checkpoint_folder = parent_folder + "checkpoint/"

        # Create Folders
        for d in [self.summary_folder, self.results_folder, self.results_folder, self.checkpoint_folder]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Save flags
        FLAGS.append_flags_into_file(parent_folder + "flags.txt")

        # Model checkpoint manager
        self.checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_folder,
                                                   filename='{epoch}',
                                                   # monitor="wasserstein_total",
                                                   save_top_k=-1,
                                                   )

        # Logger
        # tb_logger = pl_loggers.TensorBoardLogger(self.summary_folder)
        csv_logger = pl_loggers.CSVLogger(self.summary_folder, version=0)

        if FLAGS.restore or FLAGS.predict:
            resume_from_checkpoint = self.checkpoint_folder + f"epoch={FLAGS.restore_epoch}.ckpt"
        else:
            resume_from_checkpoint = None

        if FLAGS.gpus > 1:
            self.trainer = Trainer(
                callbacks=[self.checkpoint_callback],
                resume_from_checkpoint=resume_from_checkpoint,
                gpus=FLAGS.gpus,
                max_epochs=FLAGS.num_epochs * FLAGS.num_iter * 2,
                check_val_every_n_epoch=FLAGS.eval_frequency,
                # logger=[tb_logger, csv_logger],
                logger=csv_logger,
                strategy="ddp",
                log_every_n_steps=2
                # stochastic_weight_avg=True,
            )
        else:
            self.trainer = Trainer(
                callbacks=[self.checkpoint_callback],
                resume_from_checkpoint=resume_from_checkpoint,
                gpus=FLAGS.gpus,
                max_epochs=FLAGS.num_epochs * FLAGS.num_iter * 2,
                check_val_every_n_epoch=FLAGS.eval_frequency,
                # logger=[tb_logger, csv_logger],
                logger=csv_logger,
                log_every_n_steps=2
                # stochastic_weight_avg=True,
            )
        # Data
        self.prior: BaseDataGenerator = datasets_dict[FLAGS.prior]()
        self.data: BaseDataGenerator = datasets_dict[FLAGS.dataset](self.prior)

        # Model
        if FLAGS.predict or FLAGS.restore:
            self.model = Model.load_from_checkpoint(self.checkpoint_folder + f"epoch={FLAGS.restore_epoch}.ckpt")
        else:
            self.model = Model(self.data.observed_dims, not(FLAGS.restore or FLAGS.predict), True,
                               self.results_folder)

        print(f"DIR: {parent_folder}")

    def run(self):
        if not FLAGS.predict:
            self.trainer.fit(self.model, self.data)
        else:
            self.trainer.validate(self.model, self.data)
