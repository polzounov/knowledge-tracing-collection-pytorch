import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class EmbeddingDKT(Module):
    """
    Deep Knowledge Tracing Model.

    Args:
        num_q: the total number of the questions (Knowledge Components) in the given dataset
        emb_size: the dimension of the embedding vectors in this model
        hidden_size: the dimension of the hidden vectors in this model
    """

    def __init__(self, num_q, emb_size, hidden_size):
        super().__init__()

        # Initializing parameters
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        # Define an embedding layer for interaction (combined question and response)
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)

        # LSTM layer to capture sequential information
        self.lstm_layer = LSTM(self.emb_size, self.hidden_size, batch_first=True)

        # Fully connected layer for output
        self.out_layer = Linear(self.hidden_size, self.num_q)

        # Dropout layer to prevent overfitting
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        """
        Forward pass for the model.

        Args:
            q: the question (Knowledge Component) sequence with the size of [batch_size, n, emb_size]
            r: the response sequence with the size of [batch_size, n]

        Returns:
            y: the knowledge level about all questions (Knowledge Components)
        """
        # # With question indices we used Embeddings to get input to lstm layer
        # # For n=100 questions inccorect questions were represented by [0,99]
        # # and correct questions were represented by [100,199]
        # x = q + self.num_q * r

        # Now we have question embeddings. And we can represent correct/incorrect
        # responses by doubling input size and putting the embedding at start/end
        # For example, for embedding size of 5, where question embedding is [1,1,1,1,1]
        #   input_correct = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        #   input_wrong   = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        x = torch.cat((q, q), dim=-1)
        mask = torch.cat((r.unsqueeze(-1).expand_as(q), (1 - r.unsqueeze(-1)).expand_as(q)), dim=2)
        x = x * mask
        # x is shape of [batch_size, n, emb_size*2]

        h, _ = self.lstm_layer(x)
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

        # # ==============================================================================
        # TEST EMBEDDING MASKING
        # import torch
        # # Initialize the model:
        # # Generate a random question sequence:
        # q = [
        #     [[1, 2, 3],[4, 5, 6],[7, 8, 9]],
        #     [[1, 1, 1],[2, 2, 2],[3, 3, 3]],
        # ]
        # q = torch.tensor(q).float()  # Batch size of 2, 3 questions, embedding size of 3
        # r = [
        #     [0, 1, 0],
        #     [1, 0, 1],

        # ]
        # r = torch.tensor(r).float()  # Batch size of 2, 3 questions

        # # Double the embedding size for each question
        # x = torch.cat((q, q), dim=-1)
        # # Create a 2 masks with the same shape as `q` first being 1s and second being 0s
        # # is correct, and first being 0s and second being 1s if incorrect
        # mask = torch.cat((r.unsqueeze(-1).expand_as(q), (1 - r.unsqueeze(-1)).expand_as(q)), dim=2)

        # # Multiply the mask with `x` to get the desired embeddings for correct and incorrect answers
        # x = x * mask

        # # Print the output:
        # print(x)
        # # ==============================================================================

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        """
        Training loop for the model.

        Args:
            train_loader: the PyTorch DataLoader instance for training data
            test_loader: the PyTorch DataLoader instance for test data
            num_epochs: the number of epochs
            opt: the optimizer to train this model
            ckpt_path: the path to save this model's parameters
        """
        # Lists to store AUCs and average losses for each epoch
        aucs = []
        loss_means = []

        # Track maximum AUC
        max_auc = 0

        # Loop through each epoch
        for i in range(1, num_epochs + 1):
            loss_mean = []

            # Training loop
            for data in train_loader:
                # q = question sequence, r = response sequence, qshft = question sequence shifted by 1?????
                q, r, qshft, rshft, m = data

                # Set the model to training mode
                self.train()

                # Forward pass
                y = self(q.long(), r.long())

                # Mask to only consider specific questions
                y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                # Mask y and target tensors to consider only valid entries
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)

                # Reset gradients
                opt.zero_grad()

                # Calculate loss
                loss = binary_cross_entropy(y, t)

                # Backward pass
                loss.backward()

                # Update weights
                opt.step()

                # Store loss
                loss_mean.append(loss.detach().cpu().numpy())

            # Evaluation loop
            with torch.no_grad():
                for data in test_loader:
                    q, r, qshft, rshft, m = data

                    # Set the model to evaluation mode
                    self.eval()

                    # Forward pass
                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshft.long(), self.num_q)).sum(-1)

                    # Mask y and target tensors to consider only valid entries
                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    # Compute AUC
                    auc = metrics.roc_auc_score(y_true=t.numpy(), y_score=y.numpy())

                    # Calculate mean loss
                    loss_mean = np.mean(loss_mean)

                    # Display results
                    print("Epoch: {},   AUC: {},   Loss Mean: {}".format(i, auc, loss_mean))

                    # Save the model if it achieves the best AUC
                    if auc > max_auc:
                        torch.save(self.state_dict(), os.path.join(ckpt_path, "model.ckpt"))
                        max_auc = auc

                    # Update lists
                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means
