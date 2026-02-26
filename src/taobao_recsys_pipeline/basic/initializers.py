import torch


class RandomNormal:
    """Returns an embedding initialized with a normal distribution.

    Args:
        mean (float): the mean of the normal distribution
        std (float): the standard deviation of the normal distribution
    """

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim, padding_idx=None):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class RandomUniform:
    """Returns an embedding initialized with a uniform distribution.

    Args:
        minval (float): Lower bound of the range of random values of the uniform distribution.
        maxval (float): Upper bound of the range of random values of the uniform distribution.
    """

    def __init__(self, minval=0.0, maxval=1.0):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, vocab_size, embed_dim, padding_idx=None):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.uniform_(embed.weight, self.minval, self.maxval)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class XavierNormal:
    """Returns an embedding initialized with  the method described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    Args:
        gain (float): stddev = gain*sqrt(2 / (fan_in + fan_out))
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim, padding_idx=None):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.xavier_normal_(embed.weight, self.gain)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed


class XavierUniform:
    """Returns an embedding initialized with the method described in
    `Understanding the difficulty of training deep feedforward neural networks`
    - Glorot, X. & Bengio, Y. (2010), using a uniform distribution.

    Args:
        gain (float): stddev = gain*sqrt(6 / (fan_in + fan_out))
    """

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim, padding_idx=None):
        embed = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        torch.nn.init.xavier_uniform_(embed.weight, self.gain)
        if padding_idx is not None:
            torch.nn.init.zeros_(embed.weight[padding_idx])
        return embed
