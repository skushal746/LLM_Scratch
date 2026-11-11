"""A minimal decoder-only (GPT-style) transformer language model.

The model reads a sequence of token IDs and, for every position, predicts a
probability distribution over the next token. Training teaches it those
predictions; generation repeatedly samples from them to produce new text.

Overall structure (outermost to innermost):

    GPT
     ├── token embedding + position embedding
     ├── dropout
     ├── Block x n_layer
     │     ├── LayerNorm -> CausalSelfAttention -> residual add
     │     └── LayerNorm -> FeedForward         -> residual add
     ├── final LayerNorm
     └── linear head (hidden size -> vocab size logits)

Tensor shape notation used in comments:
    B = batch size (how many sequences are processed in parallel)
    T = sequence length (number of tokens in each sequence, at most block_size)
    C = embedding size (n_embd), the width of the hidden vectors
    nh = number of attention heads
    hs = head size (C // nh), the width of each head's slice of C
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position can only see earlier positions.

    Each token builds three vectors:
      - query (q): "what am I looking for?"
      - key   (k): "what do I contain?"
      - value (v): "what information do I pass along if someone attends to me?"

    A token's output is a weighted average of the values of the tokens it can see,
    where the weights come from how well its query matches each key. "Causal" means
    a token can only look at itself and tokens before it, never at future tokens.
    This is what makes the model usable for next-token prediction.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        # The embedding is split evenly across heads, so it must divide evenly.
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = n_embd // n_head  # hs: width of each head

        # A single linear layer computes q, k and v for all heads at once.
        # Its output is 3*C wide: C for queries, C for keys, C for values.
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        # After the heads run in parallel, their outputs are concatenated (back to C)
        # and mixed together by this final projection.
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)   # randomly zeros attention weights during training
        self.resid_drop = nn.Dropout(dropout)  # randomly zeros the output during training

        # Causal mask: a lower-triangular matrix of 1s. Entry [i, j] is 1 when
        # position i may attend to position j (j <= i) and 0 otherwise.
        # It is stored as a buffer so it moves to the GPU with the model but is
        # not treated as a learnable parameter.
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -- one C-dimensional vector per token, per sequence.
        B, T, C = x.shape

        # Compute queries, keys and values in one matrix multiply, then split
        # the 3C-wide result into three (B, T, C) tensors.
        q, k, v = self.qkv(x).split(C, dim=2)

        # Give each head its own slice of the embedding. Then move the head
        # dimension forward so each head can be processed independently:
        #   (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores: how strongly each query matches each key.
        #   q @ k^T -> (B, nh, T, T), where entry [i, j] compares token i with token j.
        # Dividing by sqrt(hs) keeps the scores from growing with head size, which
        # would otherwise make the softmax too sharp and gradients too small.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal masking: wherever the mask is 0 (a future token), set the score to
        # -inf. After softmax, -inf becomes a weight of exactly 0, so the token
        # cannot look ahead.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Turn each row of scores into weights that sum to 1 (a probability
        # distribution over the visible tokens), then apply dropout to the weights.
        att = self.attn_drop(F.softmax(att, dim=-1))

        # Weighted sum of values: each token's new vector is a mix of the value
        # vectors of the tokens it attends to.
        #   (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v  # (B, nh, T, hs)

        # Undo the head reshaping and concatenate all heads back into one C-wide vector:
        #   (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        # .contiguous() rearranges memory so .view() can reshape it safely.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection and dropout. Output shape: (B, T, C).
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP that expands and then projects back to n_embd.

    Attention lets tokens exchange information; this block then lets each token
    process that information independently. It widens the vector 4x, applies a
    non-linearity, and shrinks it back. Without the non-linearity, stacking
    linear layers would collapse into a single linear map, which is far less expressive.
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand: C -> 4C
            nn.GELU(),                       # smooth non-linearity (common in GPT-style models)
            nn.Linear(4 * n_embd, n_embd),  # contract: 4C -> C
            nn.Dropout(dropout),             # regularization applied to the block's output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, T, C). Each token is transformed on its own.
        return self.net(x)


class Block(nn.Module):
    """One transformer layer: attention followed by a feed-forward network.

    Two design choices here:
      - Pre-LayerNorm: LayerNorm is applied to the input of each sub-layer, before
        attention or the MLP. This keeps training stable.
      - Residual connections: each sub-layer's output is added to its input
        (x = x + ...). This gives gradients a direct path through the network and
        lets each layer learn a small adjustment rather than a full rewrite.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)   # normalizes each token's vector before attention
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)   # normalizes each token's vector before the MLP
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C). Shape is unchanged through the block.
        # Step 1: tokens exchange information with earlier tokens.
        x = x + self.attn(self.ln1(x))
        # Step 2: each token is transformed independently.
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full language model: embeddings, a stack of Blocks, and an output head."""

    def __init__(
        self,
        vocab_size: int,          # number of distinct tokens (65 for character-level Shakespeare)
        block_size: int = 128,    # maximum context length in tokens
        n_layer: int = 4,         # number of Blocks stacked on top of each other
        n_head: int = 4,          # attention heads per Block
        n_embd: int = 128,        # width of the hidden vectors (C)
        dropout: float = 0.1,     # probability of dropping activations during training
    ):
        super().__init__()
        self.block_size = block_size

        # Token embedding: a lookup table with one learned C-dimensional vector per token ID.
        # Input IDs (B, T) become vectors (B, T, C).
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # Position embedding: one learned vector per position (0 .. block_size-1).
        # Attention is order-blind on its own, so this tells the model where each token sits.
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # The stack of transformer layers. nn.Sequential runs them in order.
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        # Final LayerNorm applied to the top of the stack before the output head.
        self.ln_f = nn.LayerNorm(n_embd)
        # Output head: maps each C-dimensional vector to one score (logit) per vocabulary token.
        # No bias, since the LayerNorm before it already centers the values.
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Apply the custom initialization below to every submodule.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Start weights from small random values (std 0.02) and biases at zero.
        # Small starting weights keep early activations and logits in a reasonable range,
        # which makes the first training steps stable.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """Run the model.

        idx:     (B, T) integer token IDs for the input sequences.
        targets: (B, T) the correct next token at each position, or None when generating.

        Returns (logits, loss):
            logits: (B, T, vocab_size) unnormalized scores for the next token at each position.
            loss:   a single number (average cross-entropy), or None if no targets were given.
        """
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"

        # Combine what each token is (token embedding) with where it is (position embedding).
        # positions is [0, 1, ..., T-1]; broadcasting adds the same position vectors to every sequence.
        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(positions))  # (B, T, C)

        # Pass through all transformer blocks. Shape stays (B, T, C).
        x = self.blocks(x)

        # Normalize, then project each position's vector to one score per vocabulary token.
        logits = self.head(self.ln_f(x))  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Cross-entropy measures how much probability the model gave to the correct
            # next token. Lower is better. The (B, T) grid is flattened to one long list of
            # B*T predictions because F.cross_entropy expects (N, classes) and (N,) shapes.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()  # generation never trains the model, so skip gradient tracking to save memory and time
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend idx (shape (B, T)) by max_new_tokens tokens.

        Each step: predict a distribution for the next token, sample one token from it,
        append it to the sequence, and repeat. The output grows by one token per loop.
        """
        for _ in range(max_new_tokens):
            # The model can only handle block_size tokens at once, so keep only the most recent ones.
            idx_cond = idx[:, -self.block_size:]  # the model only sees the last block_size tokens

            # Run the model and keep only the prediction at the final position: the next token.
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)
            # temperature < 1 sharpens the distribution (more predictable text);
            # temperature > 1 flattens it (more random text).

            if top_k is not None:
                # Top-k sampling: keep only the k highest-scoring tokens and rule out the rest.
                # This stops the model from occasionally picking very unlikely tokens.
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")  # -inf becomes probability 0 after softmax

            # Convert scores to probabilities, then draw one token at random according to them.
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append the new token so it becomes part of the context for the next step.
            idx = torch.cat((idx, next_id), dim=1)
        return idx
