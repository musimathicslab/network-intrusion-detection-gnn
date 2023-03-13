import numpy as np
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn.conv import GCN2Conv


class NodeClassificator(nn.Module):
    def __init__(self, dataset, num_classes, num_convs: int = 8, hid: int = 128, dropout: float = 0.5,
                 alpha: float = 0.5, theta: float = 0.7):
        super(NodeClassificator, self).__init__()
        self.hid = hid
        self.num_layers = num_convs

        self.lin1 = nn.Linear(dataset.num_node_features, hid)

        self.convs = nn.ModuleList()
        for layer_index in range(num_convs):
            conv = GCN2Conv(
                channels=hid,
                alpha=alpha,
                theta=theta,
                layer=layer_index + 1,
                normalize=True
            )
            self.convs.append(conv)

        self.norm = LayerNorm(hid, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(hid, hid)
        self.fc2 = nn.Linear(hid, num_classes)

    def forward(self, x, edge_index, edge_attr):
        x = self.lin1(x)
        x0 = x

        for conv in self.convs:
            x = self.dropout(x)
            x = F.gelu(conv(x, x0, edge_index))

        x = F.gelu(self.norm(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = F.softmax(x, dim=1)

        return x


def train(model,
          train_dataloader,
          criterion,
          criterion_val,
          optimizer,
          device,
          model_path,
          logger=None,
          epochs=4,
          model_name='gnn',
          evaluation=False,
          patience=10,
          val_dataloader=None):
    # Print the header of the result table
    logger.info(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    logger.info("-" * 70)
    trigger_times = 0
    last_loss = np.inf
    best_model = None
    for epoch_i in range(epochs):
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode [IT'S JUST A FLAG]
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            # Load batch to GPU
            batch = batch.to(device)

            # Compute loss and accumulate the loss values
            output = model(batch.x, batch.edge_index, batch.edge_attr)
            target = batch.y
            loss = criterion(output, target)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()
            # Update parameters and the learning rate
            optimizer.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 500 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                logger.info(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | "
                    f"{time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        logger.info("-" * 70)

        if evaluation:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader, criterion_val, device)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            logger.info(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} |"
                f" {time_elapsed:^9.2f}")
            logger.info("-" * 70)

            # Early stopping
            if val_loss > last_loss:
                trigger_times += 1
            else:
                last_loss = val_loss
                best_model = model.state_dict()
                trigger_times = 0

            if trigger_times >= patience:
                torch.save(best_model, os.path.join(model_path, f'{model_name}.h5'))
                break

        else:
            # Early stopping
            if avg_train_loss > last_loss:
                trigger_times += 1
            else:
                last_loss = avg_train_loss
                best_model = model.state_dict()
                trigger_times = 0

            if trigger_times >= patience:
                torch.save(best_model, os.path.join(model_path, f'{model_name}.h5'))
                break

    torch.save(model.state_dict() if best_model is None else best_model, os.path.join(model_path, f'{model_name}.h5'))
    logger.info("\nTraining complete!")


def evaluate(model,
             val_dataloader,
             criterion,
             device):
    # Put the model into the evaluation mode. The dropout layers are disabled during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        batch = batch.to(device)

        # Compute logits
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.edge_attr)

            # Compute loss
            loss = criterion(output, batch.y)
            val_loss.append(loss.item())

            # Get the predictions
            probs = F.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            # Calculate the accuracy rate
            accuracy = (preds == batch.y).cpu().numpy().mean() * 100

            val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def predict(model,
            test_dataloader,
            device):
    # Put the model into the evaluation mode. The dropout layers are disabled during the test time.
    model.eval()
    # Init outputs
    outputs = []
    y_true = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        #  Load batch to GPU
        batch = batch.to(device)
        # Compute logits
        with torch.no_grad():
            output = model(batch.x, batch.edge_index, batch.edge_attr)
        y_true.append(batch.y)
        outputs.append(output)


    # Concatenate logits from each batch
    outputs = torch.cat(outputs, dim=0)
    y_true = torch.cat(y_true, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(outputs, dim=1).cpu().numpy()
    y_pred = np.argmax(probs, axis=1)
    y_true = y_true.cpu().numpy()

    return y_true, y_pred
