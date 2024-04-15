#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2024.

from opt_marepo import get_opts
from marepo.marepo_trainer import TrainerMarepoTransformer
import sys
import time
import logging

# pytorch-lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

_logger = logging.getLogger(__name__)

if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)
    options = get_opts()
    MarepoFormer = TrainerMarepoTransformer(options)

    if options.use_half:
        precision="16-mixed"
    else:
        precision=32

    class MyProgressBar(TQDMProgressBar):
        def init_validation_tqdm(self):
            bar = super().init_validation_tqdm()
            if not sys.stdout.isatty():
                bar.disable = True
            return bar

        def init_predict_tqdm(self):
            bar = super().init_predict_tqdm()
            if not sys.stdout.isatty():
                bar.disable = True
            return bar

        def init_test_tqdm(self):
            bar = super().init_test_tqdm()
            if not sys.stdout.isatty():
                bar.disable = True
            return bar

    callbacks = [MyProgressBar(refresh_rate=10)]

    trainer = Trainer(max_epochs=options.epochs,
                      check_val_every_n_epoch=options.check_val_every_n_epoch,
                      callbacks=callbacks,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=options.num_gpus, # hparams.num_gpus,
                      strategy=DDPStrategy(find_unused_parameters=True) if options.num_gpus > 1 else "auto",
                      num_sanity_val_steps= options.num_sanity_val_steps, # sanity check iter at beginning of the training, 0 is none, -1 is all val data
                      precision=precision,
                      # deterministic=True # try to be deterministic for unit tests
                      )

    training_start = time.time()
    trainer.fit(MarepoFormer)
    end_time = time.time()

    _logger.info(f'Done without errors. '
                 f'Total time: {end_time - training_start:.1f} seconds.')
