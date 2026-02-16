import torch
import torch.nn as nn

def save_checkpoint(model: nn.Module, optimizer: torch.optim, epoch: int, loss: float) -> None:
    """
    Save the current status of model and optimizer

    :param model: The pytorch model being trained
    :param optimizer: The pytorch optimizer used for training
    :param epoch: Which epoch is this checkpoint from
    :param loss: Loss of current point(I like to use val loss)
    """
    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    filename = f'checkpoint_{epoch}_{loss}.pth.tar'
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved at epoch {epoch} as {filename}')

def load_checkpoint(model: nn.Module, optimizer: torch.optim, filename: str) -> int:
    """
    Load the pytorch model and optimizer from checkpoint. 
    This changes model and optimizer in place so no need to return them.

    :param model: PyTorch model we're loading
    :param optimizer: PyTorch optimizer we're loading
    :param filename: File path to the checkpoint
    :return: Epoch where this checkpoint is from
    """
    checkpoint = torch.load(filename, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from epoch {epoch} with loss {loss}')
    return epoch