import warnings
import torch
import math
import sys
from tqdm import tqdm, trange


class Trainer:
    """
    Trainer class that eases the training of a PyTorch model.
    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    criterion : torch.nn.Module
        Loss function criterion.
    optimizer : torch.optim
        Optimizer to perform the parameters update.
    epochs : int
        The total number of iterations of all the training 
        data in one cycle for training the model.
    scaler : torch.cuda.amp
        The parameter can be used to normalize PyTorch Tensors 
        using native functions more detail:
        https://pytorch.org/docs/stable/index.html.
    lr_scheduler : torch.optim.lr_scheduler
        A predefined framework that adjusts the learning rate 
        between epochs or iterations as the training progresses.
    Attributes
    ----------
    train_losses_ : torch.tensor
        It is a log of train losses for each epoch step.
    val_losses_ : torch.tensor
        It is a log of validation losses for each epoch step.
    """
    def __init__(
        self, 
        model, 
        criterion, 
        optimizer,
        epochs,
        scaler=None,
        lr_scheduler=None, 
        device=None
    ):
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.device = self._get_device(device)
        self.epochs = epochs
        self.model = model.to(self.device)
        
    def fit(self, train_loader, val_loader):
        """
        Fit the model using the given loaders for the given number
        of epochs.
        
        Parameters
        ----------
        train_loader : 
        val_loader : 
        """
        # attributes  
        self.train_losses_ = torch.zeros(self.epochs)
        self.val_losses_ = torch.zeros(self.epochs)
        # ---- train process ----
        for epoch in trange(1, self.epochs + 1, desc='Traning Model on {} epochs'.format(self.epochs)):
            # train
            self._train_one_epoch(train_loader, epoch)
            # validate
            self._evaluate(val_loader, epoch)
    
    def _train_one_epoch(self, data_loader, epoch):
        self.model.train()
        losses = torch.zeros(len(data_loader))
        with tqdm(data_loader, unit=" training-batch", colour="green") as training:
            for i, (images, labels) in enumerate(training):
                training.set_description(f"Epoch {epoch}")
                images, labels = images.to(self.device), labels.to(self.device)
                # forward pass
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    preds = self.model(images)
                    loss = self.criterion(preds.float(), labels.float())
                if not math.isfinite(loss):
                    msg = f"Loss is {loss}, stopping training!"
                    warnings.warn(msg)
                    sys.exit(1)
                # remove gradient from previous passes
                self.optimizer.zero_grad()
                # backprop
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                # parameters update
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                
                training.set_postfix(loss=loss.item())
                losses[i] = loss.item()
        
            self.train_losses_[epoch - 1] = losses.mean()
    
    @torch.inference_mode()
    def _evaluate(self, data_loader, epoch):
        self.model.eval()
        losses = torch.zeros(len(data_loader))
        with tqdm(data_loader, unit=" validating-batch", colour="green") as evaluation:
            for i, (images, labels) in enumerate(evaluation):
                evaluation.set_description(f"Validation")
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.model(images)
                loss = self.criterion(preds.float(), labels.float())
                self.val_losses_[epoch - 1] = loss.item()
                evaluation.set_postfix(loss=loss.item())
                losses[i] = loss.item()

            self.val_losses_[epoch - 1] = losses.mean()

    def _get_device(self, _device):
        if _device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            msg = f"Device was automatically selected: {device}"
            warnings.warn(msg)
            return device
        return _device