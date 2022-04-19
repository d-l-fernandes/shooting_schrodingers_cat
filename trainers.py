import os
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
from pathlib import Path
import time
from typing import Optional

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
flags.DEFINE_integer("restore_epoch", -1, "Which epoch to restore.")

flags.DEFINE_integer("gpus", 1, "Number of GPUs to use",
                     lower_bound=0)
flags.DEFINE_integer("eval_frequency", 50, "How often to evaluate the model.")

FLAGS = flags.FLAGS

MAX_RESTARTS = 5


class ModelTrainer:
    def __init__(self):
        # Folders
        parent_folder = "results/"
        epoch = None

        now = datetime.datetime.now()

        if not(FLAGS.restore or FLAGS.predict):
            date = now.strftime("%Y-%m-%d")
            time = now.strftime("%H:%M:%S")
            parent_folder += f"{date}/{time}/"
        else:
            date = parent_folder + FLAGS.restore_date \
                if FLAGS.restore_date != "" else sorted(Path(parent_folder).iterdir(), key=os.path.getmtime)[-1]
            parent_folder = f"{date}/"
            if not os.path.exists(parent_folder):
                raise RuntimeError(f"{parent_folder} does not exist")

            time = parent_folder + FLAGS.restore_time \
                if FLAGS.restore_time != "" else sorted(Path(parent_folder).iterdir(), key=os.path.getmtime)[-1]
            parent_folder = f"{time}/"
            if not os.path.exists(parent_folder):
                raise RuntimeError(f"{parent_folder} does not exist")

        parent_folder += f"data_{FLAGS.dataset}/"
        parent_folder += f"prior_{FLAGS.prior}/"

        parent_folder += f"solver={FLAGS.solver}_numSteps={FLAGS.num_steps}/"

        if FLAGS.do_dsb:
            parent_folder += "dsb/"

        self.summary_folder = parent_folder + "summary/"
        self.results_folder = parent_folder + "results/"
        self.checkpoint_folder = parent_folder + "checkpoint/"

        # Create Folders
        for d in [self.summary_folder, self.results_folder, self.results_folder, self.checkpoint_folder]:
            if not os.path.exists(d):
                os.makedirs(d)

        if FLAGS.restore or FLAGS.predict:
            epoch = f"epoch={FLAGS.restore_epoch}" \
                if FLAGS.restore_epoch != -1 \
                else (sorted(Path(self.checkpoint_folder).iterdir(), key=os.path.getmtime)[-1]).parts[-1]

        # Save flags
        FLAGS.append_flags_into_file(parent_folder + "flags.txt")

        # Model checkpoint manager
        self.checkpoint_callback = ModelCheckpoint(dirpath=self.checkpoint_folder,
                                                   filename='{epoch}',
                                                   save_top_k=-1,
                                                   )

        # Logger
        self.csv_logger = pl_loggers.CSVLogger(self.summary_folder, version=0)

        if FLAGS.restore or FLAGS.predict:
            resume_from_checkpoint = self.checkpoint_folder + f"{epoch}"
        else:
            resume_from_checkpoint = None


        self.define_trainer(resume_from_checkpoint)

        # Data
        self.prior: BaseDataGenerator = datasets_dict[FLAGS.prior]()
        self.data: BaseDataGenerator = datasets_dict[FLAGS.dataset](self.prior)
        self.data.setup()

        # Model
        if FLAGS.predict or FLAGS.restore:
            self.model = Model.load_from_checkpoint(self.checkpoint_folder + f"{epoch}")
        else:
            self.model = Model(self.data.observed_dims, not(FLAGS.restore or FLAGS.predict), True,
                               self.results_folder, self.data.max_diffusion)

        # Number of restarts executed
        self.cur_restart = 0

        print(f"DIR: {parent_folder}")

    def run(self):
        if not FLAGS.predict:
            try:
                self.trainer.fit(self.model, self.data)
            except OSError as e: # In case there's a Input/Output error from the cluster
                self.cur_restart += 1
                if self.cur_restart <= MAX_RESTARTS:
                    print(f"OS Error! Restarting from latest checkpoint... ({self.cur_restart}/{MAX_RESTARTS})")
                    time.sleep(30) # Wait for a bit, because usually the error lasts for a few seconds
                    self.restart_trainer()
                    self.run()
                else:
                    print(f"OS Error! MAX_RESTARTS exceeded. Stoppping process.")
                    print(e)
            except RuntimeError as e: # In case the optimizer makes parameters go to nan/inf
                self.cur_restart += 1
                if self.cur_restart <= MAX_RESTARTS:
                    print(f"Runtime Error! Restarting from latest checkpoint... ({self.cur_restart}/{MAX_RESTARTS})")
                    self.restart_trainer()
                    self.run()
                else:
                    print(f"Runtime Error! MAX_RESTARTS exceeded. Stoppping process.")
                    print(e)
        else:
            self.trainer.validate(self.model, self.data)
    
    def define_trainer(self, resume_from_checkpoint: Optional[str]=None):
        if FLAGS.gpus > 1:
            self.trainer = Trainer(
                callbacks=[self.checkpoint_callback],
                resume_from_checkpoint=resume_from_checkpoint,
                gpus=FLAGS.gpus,
                max_epochs=FLAGS.num_epochs * FLAGS.num_iter * 2,
                check_val_every_n_epoch=FLAGS.eval_frequency,
                logger=self.csv_logger,
                strategy="ddp",
                log_every_n_steps=2
            )
        else:
            self.trainer = Trainer(
                callbacks=[self.checkpoint_callback],
                resume_from_checkpoint=resume_from_checkpoint,
                gpus=FLAGS.gpus,
                max_epochs=FLAGS.num_epochs * FLAGS.num_iter * 2,
                check_val_every_n_epoch=FLAGS.eval_frequency,
                logger=self.csv_logger,
                log_every_n_steps=2,
            )
    

    def restart_trainer(self):
        # Checks if there is any checkpoint
        list_checkpoints = list(Path(self.checkpoint_folder).iterdir())
        if len(list_checkpoints) == 0:
            # If not, creates a new model
            resume_from_checkpoint = None
            self.model = Model(
                self.data.observed_dims, not(FLAGS.restore or FLAGS.predict), 
                True, self.results_folder, self.data.max_diffusion)
        else:
            # Else, gets model from latest epoch
            epoch= (sorted(
                Path(self.checkpoint_folder).iterdir(), 
                key=os.path.getmtime)[-1]).parts[-1]
            resume_from_checkpoint = self.checkpoint_folder + f"{epoch}"
            self.model = Model.load_from_checkpoint(
                self.checkpoint_folder + f"{epoch}")

        self.define_trainer(resume_from_checkpoint)