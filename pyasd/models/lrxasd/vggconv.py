"""Structurally re-parameterizable convolutions (RepVGG) for 1D/2D/3D inputs.

During training a :class:`RepVGGBlock` sums several Conv+BN branches (optionally
with anisotropic kernels such as ``(1, 5, 5)`` and ``(5, 1, 1)``). At deployment,
:meth:`RepVGGBlock.convert_to_deploy` folds all branches into one convolution, so
outputs stay the same and inference gets faster.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act: str, inplace: bool = True):
    """Build an activation module from its name.

    Args:
        act (str | nn.Module | None): One of ``'silu'``, ``'relu'``,
            ``'leaky_relu'``, ``'gelu'``, an ``nn.Module`` instance, or ``None``
            (identity).
        inplace (bool): Value set on the module's ``inplace`` attribute, if any.

    Returns:
        nn.Module: The activation module.
    """
    if act is None:
        return nn.Identity()

    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()
    elif act == 'relu':
        m = nn.ReLU()
    elif act == 'leaky_relu':
        m = nn.LeakyReLU()
    elif act == 'gelu':
        m = nn.GELU()
    elif isinstance(act, nn.Module):
        m = act
    else:
        raise RuntimeError(f"Activation '{act}' not supported")

    if hasattr(m, 'inplace'):
        m.inplace = inplace
    
    return m 

def to_tuple(value, dim):
    """Convert a kernel size to a ``dim``-tuple.

    Args:
        value (int | tuple | list): Kernel size.
        dim (int): Expected number of dimensions.

    Returns:
        tuple: ``value`` repeated ``dim`` times (if int), or ``value`` as a tuple.

    Raises:
        ValueError: If the length does not match ``dim`` or the type is unsupported.
    """
    if isinstance(value, int):
        return (value,) * dim
    elif isinstance(value, (tuple, list)):
        if len(value) != dim:
            raise ValueError(f"Kernel size {value} does not match dimension {dim}")
        return tuple(value)
    else:
        raise ValueError(f"Unsupported type for kernel size: {type(value)}")

class ConvNormLayer(nn.Module):
    """Convolution + BatchNorm (+ activation) for 1D, 2D or 3D inputs.

    Args:
        dim (int): Spatial dimensionality of the convolution (1, 2 or 3).
        ch_in (int): Number of input channels.
        ch_out (int): Number of output channels.
        kernel_size (int | tuple): Kernel size, an int or a ``dim``-tuple.
        stride (int | tuple): Convolution stride.
        padding (int | tuple, optional): Padding. Defaults to "same" padding,
            ``(k - 1) // 2`` per dimension.
        bias (bool): Whether the convolution has a bias.
        groups (int): Number of convolution groups.
        act (str, optional): Activation name, see :func:`get_activation`.
    """

    def __init__(self, dim, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, groups=1, act=None):
        super().__init__()
        
        if dim == 1:
            self.Conv = nn.Conv1d
            self.Bn = nn.BatchNorm1d
        elif dim == 2:
            self.Conv = nn.Conv2d
            self.Bn = nn.BatchNorm2d
        elif dim == 3:
            self.Conv = nn.Conv3d
            self.Bn = nn.BatchNorm3d
        else:
            raise ValueError(f"dim must be 1, 2, or 3. Got {dim}")

        self.dim = dim
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kernel_size = to_tuple(kernel_size, dim) # Force tuple
        self.stride = stride
        self.groups = groups
        self.bias = bias

        # Calculate padding for tuple kernel sizes
        # P = (K - 1) // 2 for each dimension
        if padding is None:
            self.padding = tuple((k - 1) // 2 for k in self.kernel_size)
        else:
            self.padding = to_tuple(padding, dim)

        self.conv = self.Conv(
            self.ch_in, 
            self.ch_out, 
            self.kernel_size, 
            self.stride, 
            padding=self.padding, 
            bias=self.bias,
            groups=self.groups
        )
        
        self.norm = self.Bn(self.ch_out, momentum=0.01, eps=0.001)
        self.act = get_activation(act) 
        
        self._init_weight()

    def _init_weight(self):
        """Kaiming-normal init for the conv and identity init for the BN."""
        torch.nn.init.kaiming_normal_(self.conv.weight)
        if hasattr(self.norm, 'weight') and self.norm.weight is not None:
            self.norm.weight.data.fill_(1)
        if hasattr(self.norm, 'bias') and self.norm.bias is not None:
            self.norm.bias.data.zero_()

    def convert_to_deploy(self):
        """Fuse BatchNorm into the convolution (``conv_bn_fused``), then drop ``conv``/``norm``."""
        if not hasattr(self, "conv_bn_fused"):
            self.conv_bn_fused = self.Conv(
                self.ch_in,
                self.ch_out,
                self.kernel_size,
                self.stride,
                padding=self.padding,
                groups=self.groups,
                bias=True 
            )

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        
        if hasattr(self, "conv"):
            del self.conv
        if hasattr(self, "norm"):
            del self.norm

    def get_equivalent_kernel_bias(self):
        """Return the ``(kernel, bias)`` of the single conv equivalent to Conv+BN."""
        kernel, bias = self._fuse_bn_tensor()
        return kernel, bias

    def _fuse_bn_tensor(self):
        """Fold the BN statistics into the conv weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Fused kernel and bias.
        """
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        
        std = (running_var + eps).sqrt()
        
        view_shape = [1] * len(kernel.shape)
        view_shape[0] = -1
        t = (gamma / std).view(*view_shape)
        
        return kernel * t, beta - running_mean * gamma / std

    def forward(self, x):
        """Apply Conv+BN+act (or the fused conv after deployment).

        Args:
            x (torch.Tensor): Input of shape (B, ch_in, *spatial).

        Returns:
            torch.Tensor: Output of shape (B, ch_out, *spatial').
        """
        if hasattr(self, "conv_bn_fused"):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)


class RepVGGBlock(nn.Module):
    """Multi-branch RepVGG block that can be fused into one convolution.

    The output is ``act(sum_i ConvBN_i(x) [+ x])``. After
    :meth:`convert_to_deploy`, all branches (and the identity) are merged into a
    single convolution whose kernel is the bounding box of all branch kernels.

    Args:
        dim (int): Spatial dimensionality of the convolutions (1, 2 or 3).
        ch_in (int): Number of input channels.
        ch_out (int): Number of output channels.
        kernel_sizes (int | tuple | list): Kernel size of each branch, e.g.
            ``[(3, 3, 3), (1, 3, 3), 1]`` for 3D. A single int ``k > 1`` creates
            the standard RepVGG pair ``[k, 1]``. All sizes must be odd.
        stride (int | tuple): Stride shared by all branches.
        padding: Unused. Each branch uses "same" padding.
        bias (bool): Whether the branch convolutions have a bias.
        groups (int): Number of convolution groups.
        act (str): Activation applied after the branch sum.
        use_identity (bool): Add an identity branch (only applied when
            ``stride == 1`` and ``ch_in == ch_out``).
    """

    def __init__(self, dim, ch_in, ch_out, kernel_sizes, stride=1, padding=None, bias=False, groups=1, act='relu', use_identity=False):
        super().__init__()
        
        self.dim = dim
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stride = stride
        self.groups = groups
        
        # 1. Parse kernel_sizes into a list of tuples
        if not isinstance(kernel_sizes, list):
            raw_sizes = [kernel_sizes]
        else:
            raw_sizes = kernel_sizes

        # If user provides a single main kernel > 1 (e.g. 3), we implicitly add a 1x1 branch
        # If user provides [(1,3,3)], we assume that's what they want (no auto-add).
        # To maintain standard RepVGG behavior: if input is just an int > 1, add 1.
        if isinstance(kernel_sizes, int) and kernel_sizes > 1:
            raw_sizes = [kernel_sizes, 1]
            
        self.kernel_sizes = [to_tuple(k, dim) for k in raw_sizes]

        # 2. Calculate Max Kernel Size (Bounding Box)
        # For mixed kernels like (1,3,3) and (3,1,1), max is (3,3,3)
        self.max_kernel_size = [0] * dim
        for k_tuple in self.kernel_sizes:
            for i, k_val in enumerate(k_tuple):
                if k_val % 2 == 0:
                    raise ValueError(f"RepVGG requires odd kernel sizes. Got {k_tuple}")
                self.max_kernel_size[i] = max(self.max_kernel_size[i], k_val)
        
        self.max_kernel_size = tuple(self.max_kernel_size)

        # 3. Define Conv Class
        if dim == 1:
            self.Conv = nn.Conv1d
        elif dim == 2:
            self.Conv = nn.Conv2d
        elif dim == 3:
            self.Conv = nn.Conv3d
        
        # 4. Create Branches
        self.branches = nn.ModuleList()
        for k in self.kernel_sizes:
            self.branches.append(
                ConvNormLayer(dim, ch_in, ch_out, k, stride, padding=None, bias=bias, groups=groups, act=None)
            )

        self.use_identity = (stride == 1 and ch_in == ch_out and use_identity)
        self.act = get_activation(act)

    def forward(self, x):
        """Apply the block.

        Args:
            x (torch.Tensor): Input of shape (B, ch_in, *spatial).

        Returns:
            torch.Tensor: Output of shape (B, ch_out, *spatial').
        """
        if hasattr(self, 'conv_fused'):
            return self.act(self.conv_fused(x))

        output = 0
        for branch in self.branches:
            output += branch(x)
            
        if self.use_identity:
            output += x
            
        return self.act(output)

    def convert_to_deploy(self):
        """Merge all branches (and the identity) into ``conv_fused``, then delete ``branches``."""
        if hasattr(self, 'conv_fused'):
            return
        
        # Max padding calculation
        max_padding = tuple((k - 1) // 2 for k in self.max_kernel_size)

        self.conv_fused = self.Conv(
            self.ch_in,
            self.ch_out,
            self.max_kernel_size,
            self.stride,
            padding=max_padding,
            groups=self.groups,
            bias=True 
        )

        fused_kernel = 0
        fused_bias = 0

        # Fuse Branches
        for branch in self.branches:
            k, b = branch.get_equivalent_kernel_bias()
            
            # Pass the specific tuple size of this branch to the pad function
            k_padded = self._pad_kernel_to_max(k, branch.kernel_size)
            fused_kernel += k_padded
            fused_bias += b

        # Fuse Identity
        if self.use_identity:
            input_dim = self.ch_in // self.groups
            # 1x1...x1 kernel
            ks_1 = (1,) * self.dim 
            
            id_tensor = torch.zeros(self.ch_out, input_dim, *ks_1).to(fused_kernel.device)
            for i in range(self.ch_out):
                id_tensor[i, i % input_dim, ...] = 1

            id_padded = self._pad_kernel_to_max(id_tensor, ks_1)
            fused_kernel += id_padded

        self.conv_fused.weight.data = fused_kernel
        self.conv_fused.bias.data = fused_bias

        del self.branches

    def _pad_kernel_to_max(self, kernel, current_k_tuple):
        """Zero-pad a branch kernel (centered) to ``self.max_kernel_size``.

        Args:
            kernel (torch.Tensor): Branch kernel weight.
            current_k_tuple (tuple): Kernel size of that branch, e.g. ``(d, h, w)``.

        Returns:
            torch.Tensor: Kernel padded to the maximum kernel size.
        """
        if current_k_tuple == self.max_kernel_size:
            return kernel
        
        # Calculate padding per dimension
        # F.pad needs reverse order: [last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right]
        pad_arg = []
        
        # Iterate dimensions in reverse (standard for F.pad)
        for i in range(self.dim - 1, -1, -1):
            k_curr = current_k_tuple[i]
            k_max = self.max_kernel_size[i]
            
            diff = k_max - k_curr
            pad_side = diff // 2
            pad_arg.extend([pad_side, pad_side])
            
        return F.pad(kernel, pad_arg)


if __name__ == "__main__":
    print("----------------------------------------------------------------")
    print("Running RepVGG Block Fusion Verification (Tuple/Anisotropic Support)...")
    print("----------------------------------------------------------------")
    
    # Test 3D Anisotropic Case
    dim = 3
    ch_in = 4
    ch_out = 4
    
    # Scenario: One branch is (1,3,3), one is (3,3,3), plus identity
    # The 'max' kernel will be (3,3,3)
    kernel_sizes = [(1,3,3), (3, 1, 1)] 
    
    print(f"\n[Testing {dim}D Anisotropic Kernels: {kernel_sizes}]")
    
    model = RepVGGBlock(
        dim=dim, 
        ch_in=ch_in, 
        ch_out=ch_out, 
        kernel_sizes=kernel_sizes, 
        stride=1,
        groups=1,
        act='relu'
    )
    
    input_tensor = torch.randn(2, ch_in, 5, 10, 10) # D, H, W
    
    model.eval()
    with torch.no_grad():
        y_origin = model(input_tensor)
        
    model.convert_to_deploy()
    
    with torch.no_grad():
        y_fused = model(input_tensor)
        
    diff = (y_origin - y_fused).abs().max()
    print(f"  Max Error: {diff.item():.8f}")
    
    if diff < 1e-5:
        print("  >> Anisotropic Test PASSED ✅")
    else:
        print("  >> Anisotropic Test FAILED ❌")