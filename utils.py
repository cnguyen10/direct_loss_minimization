import torch
import typing

def grad_estimation(x: torch.Tensor, y: torch.Tensor, net: torch.nn.Module, num_classes: int, epsilon: float=-0.1) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    '''Estimate the gradient of 0-1 loss

    Args:
        x: a batch of input data
        y: a list of labels. The labels are index-based, not one-hot vectors
        net: the network defined in PyTorch nn module
        num_classes: the number of classes to convert from index-based label to one-hot vector
        epsilon: the parameter for gradient estimation. I tried with positive value,
            but got gradient vanishing. Please use the negative value closed to 0.

    Returns: a tuple consisting of:
        loss: the average 0-1 loss value of the mini-batch
        grad: the gradient w.r.t. parameters of net average on the mini-batch
    '''
    # calculate the softmax of the mini-batch
    logits = net.forward(input=x) # (batch_size, num_classes)

    # get model's prediction
    y_model = logits.argmax(dim=1) # (batch_size,)

    # calculate the labels perturbed by 0-1 loss
    y_direct = torch.argmax(
        input=logits + epsilon * (1 - torch.nn.functional.one_hot(input=y, num_classes=num_classes)),
        dim=1
    ) # (batch_size,)

    # region GRADIENT OF SCORE FUNCTION
    F_direct = torch.mean(
        input=logits[torch.arange(start=0, end=y_direct.numel(), step=1), y_direct],
        dim=0
    )
    dF_direct = torch.autograd.grad(
        outputs=F_direct,
        inputs=net.parameters(),
        retain_graph=True
    )

    F_model = torch.mean(
        input=logits[torch.arange(start=0, end=y_model.numel(), step=1), y_model],
        dim=0
    )
    dF_model = torch.autograd.grad(
        outputs=F_model,
        inputs=net.parameters(),
        retain_graph=False
    )
    # endregion

    # region LOSS and approximated GRADIENT
    with torch.no_grad():
        loss = torch.mean(input=1 - (y_model == y).float(), dim=0)

        grad = [None] * len(dF_direct)
        for i in range(len(grad)):
            grad[i] = (dF_direct[i] - dF_model[i]) / epsilon
    # endregion

    return loss, grad