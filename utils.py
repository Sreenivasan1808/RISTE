import torch.nn as nn
import torch

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    # pe = torch.zeros(d_model, height, width)
    # # Each dimension use half of d_model
    # d_model = int(d_model / 2)
    # div_term = torch.exp(torch.arange(0., d_model, 2) *
    #                      -(math.log(10000.0) / d_model))
    # pos_w = torch.arange(0., width).unsqueeze(1)
    # pos_h = torch.arange(0., height).unsqueeze(1)
    # pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    # pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    # pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    # pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    # return pe
    pe = np.zeros((height, width, d_model))
    for x in range(height):
        for y in range(width):
            for i in range(d_model // 2):
                pe[x, y, 2 * i] = np.sin(x / (10000 ** (2 * i / d_model)))
                pe[x, y, 2 * i + 1] = np.cos(y / (10000 ** (2 * i / d_model)))

    return pe


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        self.d_model = d_model

        # Create a positional encoding matrix
        pe = torch.zeros(height, width, d_model)
        y_pos = torch.arange(height, dtype=torch.float).unsqueeze(1)  # Shape: (height, 1)
        x_pos = torch.arange(width, dtype=torch.float).unsqueeze(0)   # Shape: (1, width)

        # Calculate the positional encodings
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # Shape: (d_model/2)

        # Apply sine and cosine to even and odd indices
        pe[:, :, 0::2] = torch.sin(y_pos * div_term.unsqueeze(0))  # Shape: (height, width, d_model/2)
        pe[:, :, 1::2] = torch.cos(x_pos * div_term)              # Shape: (height, width, d_model/2)

        # Reshape to (height * width, d_model)
        pe = pe.view(height * width, d_model)
        self.register_buffer('pe', pe)


    def forward(self, x):
        # Add positional encoding to the input tensor
        return x + self.pe[:x.size(1), :].unsqueeze(0) 



# def encode_labels(labels, max_length=None):
#     """
#     Convert string labels to numerical tensor format.
    
#     Args:
#         labels (list of str): List of string labels to encode.
#         max_length (int, optional): Maximum length for padding. If None, it will be determined from the labels.

#     Returns:
#         torch.Tensor: A tensor of shape (batch_size, max_length) containing the encoded labels.
#     """
#     # Create a mapping from characters to indices
#     char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(set('abcdefghijklmnopqrstuvwxyz0123456789 ')))}
#     char_to_index['<blank>'] = 0  # Add a blank token for CTC loss

#     # Encode the labels
#     encoded_labels = []
#     for label in labels:
#         encoded_label = [char_to_index[char] for char in label if char in char_to_index]
#         encoded_labels.append(encoded_label)

#     # Determine the maximum length if not provided
#     if max_length is None:
#         max_length = max(len(seq) for seq in encoded_labels)

#     # Pad sequences to the maximum length
#     padded_labels = [seq + [0] * (max_length - len(seq)) for seq in encoded_labels]  # 0 for padding

#     return torch.tensor(padded_labels, dtype=torch.long)



def encode_labels_with_positions(labels, max_length=None):
    """
    Convert string labels to numerical tensor format, including positional information.

    Args:
        labels (list of str): List of string labels to encode.
        max_length (int, optional): Maximum length for padding. If None, it will be determined from the labels.

    Returns:
        dict: A dictionary with two tensors:
              - 'char_classes': Encoded character classes (batch_size, max_length).
              - 'char_positions': Relative positions of characters (batch_size, max_length).
    """
    # Create a mapping from characters to indices
    char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(set('\t\n !"#$%&\'()*+,-./0123456789:;<=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz|~©®°·€←→、，：')))}
    char_to_index['<blank>'] = 0  # Add a blank token for CTC loss

    # Encode the labels and positions
    encoded_labels = []
    encoded_positions = []
    for label in labels:
        encoded_label = [char_to_index[char] for char in label if char in char_to_index]
        encoded_position = list(range(len(encoded_label)))
        encoded_labels.append(encoded_label)
        encoded_positions.append(encoded_position)

    # Determine the maximum length if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in encoded_labels)

    # Pad sequences to the maximum length
    padded_labels = [seq + [0] * (max_length - len(seq)) for seq in encoded_labels]  # 0 for padding
    padded_positions = [seq + [-1] * (max_length - len(seq)) for seq in encoded_positions]  # -1 for padding positions

    return {
        'char_classes': torch.tensor(padded_labels, dtype=torch.long),
        'char_positions': torch.tensor(padded_positions, dtype=torch.long)
    }


if __name__ == "__main__":
    labels = encode_labels_with_positions(["hello", "VANAKAM"])
    print(labels)