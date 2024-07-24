from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

import torch
from torch.utils.data import Dataset
import numpy as np


import copy

import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.stats import multivariate_normal


class SIRModel(Dataset):
    def __init__(self, size_list, beta, gamma, steps, dt, interval, sigma, rho):
        """
        Initialize the SIR model dataset.
        
        :param size_list: List of initial state sizes.
        :param beta: Infection rate.
        :param gamma: Recovery rate.
        :param steps: Number of steps to run (including the starting point).
        :param dt: Step size.
        :param interval: Sampling interval.
        :param sigma: Standard deviation of noise.
        :param rho: Correlation coefficient of noise.
        """
        self.size_list = size_list
        self.beta, self.gamma = beta, gamma
        self.sigma, self.rho = sigma, rho
        self.steps = steps
        self.dt = dt
        self.interval = interval
        self.init_total_number = np.sum(size_list)

        #self.data = self.simulate_multiseries(size_list)
        self.prior = multivariate_normal(mean=np.zeros(2), cov=np.array([[1, rho], [rho, 1]]))

        self.sir_input, self.sir_output = self._simulate_multiseries()

    def perturb(self, S, I):
        """
        Add observational noise to the macro states S and I.
        
        :param S: Susceptible population.
        :param I: Infected population.
        :return: Observed states with noise.
        """
        noise_S = self.prior.rvs(size=1) * self.sigma
        noise_I = self.prior.rvs(size=1) * self.sigma
        S_obs0 = np.expand_dims(S + noise_S[0], axis=0)
        S_obs1 = np.expand_dims(S + noise_S[1], axis=0)
        I_obs0 = np.expand_dims(I + noise_I[0], axis=0)
        I_obs1 = np.expand_dims(I + noise_I[1], axis=0)
        SI_obs = np.concatenate((S_obs0, I_obs0, S_obs1, I_obs1), 0)
        return SI_obs
    
    def simulate_oneserie(self, S, I):
        """
        Simulate a single time series from a specific starting point.
        
        :param S: Initial susceptible population (as a ratio).
        :param I: Initial infected population (as a ratio).
        :return: Time series data.
        """
        sir_data = []
        for k in range(self.steps):
            if k % self.interval == 0:
                SI_obs = self.perturb(S, I)
                sir_data.append(SI_obs)
                
            new_infected = self.beta * S * I 
            new_recovered = self.gamma * I
            S = S - new_infected * self.dt
            I = I + (new_infected - new_recovered) * self.dt
        return np.array(sir_data)

    def _simulate_multiseries(self):
        """
        Simulate multiple time series from various starting points to create the main dataset.
        
        :return: sir_input and sir_output arrays.
        """
        num_obs = int(self.steps / self.interval)
        sir_data_all = np.zeros([self.init_total_number, num_obs, 4])
        num_strip = len(self.size_list)
        frac = 1 / num_strip
        
        for strip in range(num_strip):
            sir_data_part = np.zeros([self.size_list[strip], num_obs, 4])
            boundary_left = strip * frac
            boundary_right = boundary_left + frac
            S_init = np.random.uniform(boundary_left, boundary_right, self.size_list[strip])
            I_init = []
            while len(I_init) < self.size_list[strip]:
                I = np.random.rand(1)[0]
                S = S_init[len(I_init)]
                if S + I <= 1:
                    sir_data_part[len(I_init),:,:] = self.simulate_oneserie(S, I)
                    I_init.append(I)
            size_list_cum = np.cumsum(self.size_list)
            size_list_cum = np.concatenate([[0], size_list_cum])
            sir_data_all[size_list_cum[strip]:size_list_cum[strip+1], :, :] = sir_data_part
        sir_input, sir_output = self.reshape(sir_data_all = sir_data_all)
        return sir_input, sir_output

    def reshape(self, sir_data_all):
        """
        Reshape the generated multi-time series into input and output arrays.
        
        :param sir_data_all: Array of all time series data.
        :return: sir_input and sir_output arrays.
        """
        sir_input = sir_data_all[:, :-1, :].reshape(-1, 4)
        sir_output = sir_data_all[:, 1:, :].reshape(-1, 4)
        return sir_input, sir_output

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.sir_input)

    def __getitem__(self, idx):
        """
        Return an item from the dataset.
        
        :param idx: Index of the item.
        :return: A tuple of torch.Tensor representing the input and output.
        """
        return torch.tensor(self.sir_input[idx], dtype=torch.float), torch.tensor(self.sir_output[idx], dtype=torch.float)


class SIRDataModule(LightningDataModule):
    """`LightningDataModule` for the SIR dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a `SIRDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """


        self.data_train = SIRModel(size_list=[9000], beta=1, gamma=0.5, steps=7, dt=0.01, interval=1, sigma=0.03, rho=-0.5)
        self.data_val = SIRModel(size_list=[100], beta=1, gamma=0.5, steps=7, dt=0.01, interval=1, sigma=0.03, rho=-0.5)

        # Copy the validation data to the test data
        self.data_test = copy.deepcopy(self.data_val)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    # _ = MNISTDataModule()
    sir = SIRModel(size_list=[9000], beta=1, gamma=0.5, steps=7, dt=0.01, interval=1, sigma=0.03, rho=-0.5)
    # data = sir.simulate_multiseries()
    # print(data[0].shape, data[1].shape)

    # print('debug')
