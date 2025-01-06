import torch
from utils import encode_labels_with_positions
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss functions
char_class_loss_fn = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padded indices
pos_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padded positions
max_length = 170

def test_model(test_dataloader):
    model = torch.load("./results/i2c_saved.pt", weights_only=False)
    model.eval()
    running_loss = 0.0
    total_char_correct = 0
    total_pos_correct = 0
    total_samples = 0

    for batch_idx, (images, labels) in enumerate(test_dataloader):
        images = images.to(device)
        # Encode labels with positions
        encoded_labels = encode_labels_with_positions(labels, max_length=max_length)
        encoded_labels['char_classes'] = encoded_labels['char_classes'].to(device)
        encoded_labels['char_positions'] = encoded_labels['char_positions'].to(device)
        
        char_logits, pos_logits = model(images)

        # Calculate loss
        char_class_loss = char_class_loss_fn(
            char_logits.view(-1, 102), encoded_labels['char_classes'].view(-1)
        )
        char_position_loss = pos_loss_fn(
            pos_logits.view(-1, 200), encoded_labels['char_positions'].view(-1)
        )
        total_loss = char_class_loss + char_position_loss
        running_loss += total_loss.item()

        # Calculate accuracy
        char_preds = torch.argmax(char_logits, dim=-1)  # [batch_size, max_length]
        pos_preds = torch.argmax(pos_logits, dim=-1)    # [batch_size, max_length]

        # Mask out padded positions
        char_mask = (encoded_labels['char_classes'] != 0)  # Ignore padded characters
        pos_mask = (encoded_labels['char_positions'] != -1)  # Ignore padded positions

        # Count correct predictions
        char_correct = (char_preds[char_mask] == encoded_labels['char_classes'][char_mask]).sum().item()
        pos_correct = (pos_preds[pos_mask] == encoded_labels['char_positions'][pos_mask]).sum().item()

        total_char_correct += char_correct
        total_pos_correct += pos_correct
        total_samples += char_mask.sum().item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(test_dataloader)
    char_accuracy = total_char_correct / total_samples
    pos_accuracy = total_pos_correct / total_samples

    print(f"Average Loss: {avg_loss}")
    print(f"Character Accuracy: {char_accuracy * 100:.2f}%")
    print(f"Position Accuracy: {pos_accuracy * 100:.2f}%")