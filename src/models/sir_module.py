from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import Metric
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import MeanAbsoluteError, MeanMetric, MetricCollection, MinMetric
from utils.ei import EI
# import matplotlib.pyplot as plt

MeanAbsoluteError.__name__ = "MAE"


class SIRLitModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        self.net = net

        # loss function
        self.criterion = torch.nn.L1Loss()

        # metrics
        metrics = MetricCollection([MeanAbsoluteError()])
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_ei = EI()
        self.val_ei = EI()
        self.test_ei = EI()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_mae_best = MinMetric()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x, y)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_metrics.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

        self.train_ei.reset()
        self.val_ei.reset()
        self.test_ei.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        # cal_ei: return h_t, h_t1
        x, y = batch
        y_hat, ei_items = self.forward(x, y)
        
        loss = self.criterion(y_hat, y)
        
        return loss, y_hat, y, ei_items

    # def save_and_log_figure(self, h_t, batch_idx):

    #     plt.figure()
    #     plt.scatter(h_t[:,0].detach().cpu().data.numpy, h_t[:,1].detach().cpu().data.numpy, s=1, alpha=0.25)  # 根据你的数据选择合适的参数

    #     # 保存图片并记录到AIM
    #     fig_path = f"data/figure_{batch_idx}.png"  # 定义图片保存路径
    #     plt.savefig(fig_path)
    #     plt.close()  

    #     # 使用self.log记录图片到AIM
    #     self.log("train/figure", fig_path, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets, ei_items = self.model_step(batch)
        # h_t = ei_items['h_t']
        # update and log metrics
        self.train_loss(loss)
        # if batch_idx % 100 == 0:  # 例如，每10个批次生成一次图片
        #     self.save_and_log_figure(h_t, batch_idx)
        # self.train_ei(ei_items)

        # ei, ei_term1, ei_term2 = self.train_ei(ei_items)

        metrics_dict = self.train_metrics(preds.flatten(), targets.flatten())
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # self.log("train/ei", self.train_ei, on_step=True, on_epoch=True, prog_bar=True)

        # self.log("train/ei", ei, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/ei_term1", ei_term1, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("train/ei_term2", ei_term2, on_step=True, on_epoch=True, prog_bar=True)


        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, ei_items = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)

        ei, ei_term1, ei_term2 = self.val_ei(ei_items)

        metrics_dict = self.val_metrics(preds.flatten(), targets.flatten())
        
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/ei", self.val_ei, on_step=True, on_epoch=True, prog_bar=True)

        self.log("val/ei", ei, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/ei_term1", ei_term1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/ei_term2", ei_term2, on_step=True, on_epoch=True, prog_bar=True)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    # TODO
    # def on_validation_epoch_end(self) -> None:
    #     "Lightning hook that is called when a validation epoch ends."
    #     acc = self.val_acc.compute()  # get current val acc
    #     self.val_acc_best(acc)  # update best so far val acc
    #     # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
    #     # otherwise metric would be reset by lightning after each epoch
    #     self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets, ei_items = self.model_step(batch)

        # self.test_ei(ei_items)

        ei, ei_term1, ei_term2 = self.test_ei(ei_items)

        # update and log metrics
        self.test_loss(loss)
        metrics_dict = self.test_metrics(preds.flatten(), targets.flatten())

        self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("test/ei", self.test_ei, on_step=True, on_epoch=True, prog_bar=True)

        self.log("test/ei", ei, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/ei_term1", ei_term1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/ei_term2", ei_term2, on_step=True, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SIRLitModule(None, None, None, None)
