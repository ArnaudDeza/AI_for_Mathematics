import torch
import torch.nn as nn

class LR(nn.Module):
    def __init__(self, model_args):
        """
        Logistic Regression model for sequential graph construction.

        Args:
            args: An object containing model arguments.
                args.model_args: Dictionary with architecture-related parameters:
                    - None required for logistic regression.
        """
        super(LR, self).__init__()
        # Linear layer without bias
        self.linear = nn.Linear(model_args['input_size'],1, bias=True)

    def forward(self, x):
        """
        Forward pass of the Logistic Regression model.

        Args:
            x: Input tensor of shape (batch_size, 2 * E).

        Returns:
            logit: Tensor of shape (batch_size, 1) representing the logit for the binary decision.
        """
        logit = self.linear(x)  # (batch_size, 1)
        # apply sigmoid
        logit = torch.sigmoid(logit)
        return logit