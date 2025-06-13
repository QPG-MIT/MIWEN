import torch
import torch.nn as nn
import torch.nn.functional as F


class InverseComb(nn.Module):
    """
    1) Performs an orthonormal FFT on each input row (real --> complex).
    2) Keeps (out_vector_dim+1) frequencies from DC upward, plus symmetric negative frequencies.
       Out of these (out_vector_dim+1) frequency bins, exactly one (here, the first positive frequency)
       is forced to 1, while the other out_vector_dim bins retain their original values.
       The corresponding negative frequency for the special positive frequency is also set to 1.
    3) Inverse FFT (orthonormal).
    4) Returns real(...) of the result, which remains real if input was real.

    If (out_vector_dim+1) is larger than possible (i.e. > in_time_dim//2 + 1),
    we skip filtering altogether and return the original x.
    """
    def __init__(self, in_time_dim: int, out_vector_dim: int):
        super().__init__()
        self.in_time_dim = in_time_dim
        self.out_vector_dim = out_vector_dim

        # The number of frequency components to select is out_vector_dim+1.
        self.n_components = out_vector_dim + 1

        # Maximum number of distinct "positive" frequencies for a real signal of length N
        self.max_positive_freqs = in_time_dim // 2 + 1

        # Flag to indicate if we skip filtering
        self.no_filtering = False

        # If n_components is too large, just skip all filtering
        if self.n_components > self.max_positive_freqs:
            print(
                f"[InverseComb Warning] Requested out_vector_dim+1={self.n_components} is greater "
                f"than {self.max_positive_freqs} for in_time_dim={in_time_dim}. "
                "Skipping filtering; returning original vectors."
            )
            self.no_filtering = True
            return

        # Build a boolean mask to select frequencies
        mask = torch.zeros(in_time_dim, dtype=torch.bool)

        # Select n_components frequencies from the positive side (DC upward)
        positive_indices = torch.round(
            torch.linspace(0, self.max_positive_freqs - 1, self.n_components)
        ).long()
        mask[positive_indices] = True

        # For the negative frequencies, mirror the indices (except the DC term at index 0)
        if self.n_components > 1:
            negative_indices = torch.arange(
                in_time_dim - (self.n_components - 1),
                in_time_dim,
                dtype=torch.long
            )
            mask[negative_indices] = True

        # Register the mask as a buffer so it moves with the model
        self.register_buffer('freq_mask', mask)
        # Save the index to override with a constant 1 (we use the first positive index)
        self.override_index = int(positive_indices[0].item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (batch_size, in_time_dim), real input in time domain.
        Returns:
            x_time: shape (batch_size, in_time_dim), real output in time domain.
        """
        # If filtering is disabled, return the original input.
        if self.no_filtering:
            return x

        # 1) Forward FFT (orthonormal)
        x_freq = torch.fft.fft(x, norm='ortho')  # shape (batch_size, in_time_dim), complex

        # 2) Zero out the frequencies not selected by the mask
        x_freq[:, ~self.freq_mask] = 0 + 0j

        # 3) Override the special frequency with 1.
        x_freq[:, self.override_index] = 1 + 0j

        # 4) Also override its corresponding negative frequency if it is not DC.
        if self.override_index != 0:
            neg_index = self.in_time_dim - self.override_index
            x_freq[:, neg_index] = 1 + 0j

        # 5) Inverse FFT (orthonormal)
        x_time = torch.fft.ifft(x_freq, norm='ortho')

        # 6) Return the real part of the time-domain signal
        return x_time.real
