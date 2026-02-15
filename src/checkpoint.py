def save_checkpoint(model, optimizer, epoch, loss):
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

def load_checkpoint(model, optimizer, filename) -> int:
    checkpoint = torch.load(filename, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f'Checkpoint loaded from epoch {epoch} with loss {loss}')
    return epoch