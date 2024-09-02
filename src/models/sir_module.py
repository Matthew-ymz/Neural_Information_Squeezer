from typing import Any, Dict, Tuple
import numpy as np
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import Metric
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics import MeanAbsoluteError, MeanMetric, MetricCollection, MinMetric
from utils.ei import EI
from sklearn.neighbors import KernelDensity
# import matplotlib.pyplot as plt

MeanAbsoluteError.__name__ = "MAE"

def kde_density(X):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.05, atol=0.2).fit(X.cpu().data.numpy()) #bindwidth=0.02
    log_density = kde.score_samples(X.cpu().data.numpy())
    return log_density


class NISLitModule(LightningModule):
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


class NISpLitModule(NISLitModule):
    def __init__(self):
        super().__init__()

        self.temperature = None
        self.scale = None
        self.L = None
        self.weights = torch.ones()
        self.criterion = torch.nn.L1Loss(reduction='none')
        self.mae2_w = 1

    def update_weight(self, h_t, L=1):
        samples = h_t.size(0)
        scale = h_t.size(1)
        log_density = kde_density(h_t)
        log_rho = - scale * np.log(2.0 * L) 
        logp = log_rho - log_density
        soft = nn.Softmax(dim=0)
        weights = soft(torch.tensor(logp))
        weights = weights * samples
        self.weights = weights.cuda()

    def reweight(self):
        h_t_all = self.net.encoding(self.x_t_all)
        self.update_weight(h_t_all)

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
        x, y, w = batch
        y_hat, ei_items = self.forward(x, y)
        h_t_hat = self.net.back_forward(y)
        loss1 = (self.criterion(y_hat, y).mean(axis=1) * w).mean() 
        loss2 = (self.criterion(h_t_hat, ei_items['h_t']).mean(axis=1) * w).mean()
        loss = loss1 + self.mae2_w * loss2
        return loss, y_hat, y, ei_items
    
    def training_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:

            loss, preds, targets, ei_items = self.model_step(batch)
            h_t = ei_items['h_t']

            self.train_loss(loss)

            metrics_dict = self.train_metrics(preds.flatten(), targets.flatten())
            self.log_dict(metrics_dict, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

            # self.log("train/ei", self.train_ei, on_step=True, on_epoch=True, prog_bar=True)

            # self.log("train/ei", ei, on_step=True, on_epoch=True, prog_bar=True)
            # self.log("train/ei_term1", ei_term1, on_step=True, on_epoch=True, prog_bar=True)
            # self.log("train/ei_term2", ei_term2, on_step=True, on_epoch=True, prog_bar=True)


            # return loss or backpropagation will fail
            return loss

    def train_step2(self, mae2_w, batch_size):
        self.net.train()
        start = np.random.randint(self.samp_num - batch_size)
        end = start + batch_size
        x_t, x_t1, w = self.x_t_all[start:end], self.x_t1_all[start:end], self.weights[start:end]
        x_t1_hat, ei_items = self.net(x_t, x_t1)
        h_t_hat = self.net.back_forward(x_t1)
        mae1 = (self.MAE_raw(x_t1, x_t1_hat).mean(axis=1) * w).mean() 
        mae2 = (self.MAE_raw(h_t_hat, ei_items['h_t']).mean(axis=1) * w).mean() 
        loss = mae1 + mae2_w * mae2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def training(self, T1, T_all, mae2_w, batch_size, clip=200):
        for epoch in range(T_all):
            if epoch < T1:
                self.train_loss += self.train_step(batch_size)
            else:
                self.train_loss += self.train_step2(mae2_w, batch_size)
            if epoch % clip == 0:
                self.test_loss, dei, term1, term2 = self.test_step()
                self.train_loss /= clip
                self.log(dei, term1, term2, epoch)
                self.train_loss = 0
            if epoch > T1 and epoch % 1000 == 0:
                self.reweight()
                print(self.weights[:10])


if __name__ == "__main__":
    _ = NISLitModule(None, None, None, None)
