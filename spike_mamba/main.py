from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional logging with loguru
from loguru import logger

LOGURU_AVAILABLE = True

from mamba_ssm.modules.mamba_simple import Mamba

MAMBA2_AVAILABLE = True

MAMBA_AVAILABLE = True

if LOGURU_AVAILABLE:
    logger.info("Mamba package loaded successfully")


# Type aliases for better readability
TensorType = torch.Tensor
StateDict = Dict[str, TensorType]
SpikeStates = Dict[str, StateDict]
MambaCaches = List[Optional[Any]]


@dataclass
class MambaConfig:
    """Configuration for base Mamba model"""
    d_model: int = 512
    n_layer: int = 24
    vocab_size: int = 50277
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    layer_idx: Optional[int] = None
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None


@dataclass
class SpikingMambaConfig:
    """Configuration class for SpikingMamba model"""
    # Base Mamba config
    mamba_config: MambaConfig
    
    # Spiking-specific parameters
    threshold: float = 1.0
    tau_mem: float = 20.0  # membrane time constant
    tau_syn: float = 5.0   # synaptic time constant
    reset_mode: str = "subtract"  # "subtract" or "zero"
    surrogate_grad: str = "fast_sigmoid"
    adaptive_threshold: bool = True
    refractory_period: int = 2
    spike_gate_mode: str = "soft"  # "soft" or "hard"
    membrane_decay: float = 0.9
    spike_regularization: float = 0.01
    
    # Integration modes
    spike_integration: str = "pre_post"  # "pre", "post", "pre_post", "residual"
    temporal_pooling: str = "none"  # "none", "adaptive", "learnable"
    
    # Model architecture
    use_mamba2: bool = False  # Use Mamba2 if available
    
    # Logging configuration
    enable_logging: bool = True
    log_spike_stats: bool = True
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization"""
        valid_reset_modes = ["subtract", "zero"]
        if self.reset_mode not in valid_reset_modes:
            raise ValueError(f"reset_mode must be one of {valid_reset_modes}")
        
        valid_spike_modes = ["soft", "hard"]
        if self.spike_gate_mode not in valid_spike_modes:
            raise ValueError(f"spike_gate_mode must be one of {valid_spike_modes}")
        
        valid_integration_modes = ["pre", "post", "pre_post", "residual"]
        if self.spike_integration not in valid_integration_modes:
            raise ValueError(f"spike_integration must be one of {valid_integration_modes}")
        
        valid_pooling_modes = ["none", "adaptive", "learnable"]
        if self.temporal_pooling not in valid_pooling_modes:
            raise ValueError(f"temporal_pooling must be one of {valid_pooling_modes}")


class SurrogateGradient(torch.autograd.Function):
    """Surrogate gradient function for spiking neurons with configurable slope"""
    
    @staticmethod
    def forward(ctx, input: TensorType, threshold: TensorType, slope: float = 5.0) -> TensorType:
        ctx.save_for_backward(input, threshold)
        ctx.slope = slope
        return (input >= threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output: TensorType) -> Tuple[Optional[TensorType], None, None]:
        input, threshold = ctx.saved_tensors
        slope = ctx.slope
        # Fast sigmoid surrogate gradient
        diff = slope * (input - threshold)
        grad_input = grad_output * slope * torch.sigmoid(diff) * (1 - torch.sigmoid(diff))
        return grad_input, None, None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron module with comprehensive state management"""
    
    def __init__(self, config: SpikingMambaConfig, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.threshold = config.threshold
        self.tau_mem = config.tau_mem
        self.tau_syn = config.tau_syn
        self.reset_mode = config.reset_mode
        self.adaptive_threshold = config.adaptive_threshold
        self.refractory_period = config.refractory_period
        
        # Learnable membrane dynamics
        self.beta_mem = nn.Parameter(torch.tensor(config.membrane_decay))
        self.beta_syn = nn.Parameter(torch.exp(torch.tensor(-1.0 / config.tau_syn)))
        
        # Adaptive threshold parameters
        if config.adaptive_threshold:
            self.threshold_adapt = nn.Parameter(torch.ones(1) * 0.1)
            self.threshold_scale = nn.Parameter(torch.ones(d_model))
        
        # Learnable threshold per neuron
        self.learnable_threshold = nn.Parameter(torch.ones(d_model) * config.threshold)
        
        if config.enable_logging and LOGURU_AVAILABLE:
            logger.debug(f"Initialized LIFNeuron with d_model={d_model}, threshold={config.threshold}")
    
    def _init_state_dict(self, batch_size: int, device: torch.device) -> StateDict:
        """Initialize state dictionary for LIF neurons"""
        return {
            'membrane': torch.zeros(batch_size, 1, self.d_model, device=device),
            'synaptic': torch.zeros(batch_size, 1, self.d_model, device=device),
            'threshold_adapt': torch.zeros(batch_size, 1, self.d_model, device=device),
            'refractory': torch.zeros(batch_size, 1, self.d_model, device=device),
        }
    
    def forward(self, x: TensorType, state_dict: Optional[StateDict] = None) -> Tuple[TensorType, StateDict]:
        """
        Forward pass of LIF neuron
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            state_dict: Optional state dictionary from previous timestep
            
        Returns:
            spike_train: Output spikes of shape (batch_size, seq_len, d_model)
            updated_state_dict: Updated state dictionary
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        if state_dict is None:
            state_dict = self._init_state_dict(batch_size, device)
        
        # Extract states
        membrane = state_dict['membrane']
        synaptic = state_dict['synaptic']
        threshold_adapt = state_dict['threshold_adapt']
        refractory = state_dict['refractory']
        
        spike_outputs: List[TensorType] = []
        
        for t in range(seq_len):
            current_input = x[:, t:t+1, :]
            
            # Update synaptic current
            synaptic = self.beta_syn * synaptic + current_input
            
            # Update membrane potential (only for non-refractory neurons)
            non_refractory_mask = (refractory <= 0).float()
            membrane = self.beta_mem * membrane + synaptic * non_refractory_mask
            
            # Compute adaptive threshold
            if self.adaptive_threshold:
                current_threshold = (self.learnable_threshold + 
                                   self.threshold_adapt * threshold_adapt * self.threshold_scale)
            else:
                current_threshold = self.learnable_threshold
            
            # Generate spikes using surrogate gradient
            spikes = SurrogateGradient.apply(membrane, current_threshold)
            spike_outputs.append(spikes)
            
            # Reset mechanism
            if self.reset_mode == "subtract":
                membrane = membrane - spikes * current_threshold
            elif self.reset_mode == "zero":
                membrane = membrane * (1 - spikes)
            
            # Update adaptive threshold
            if self.adaptive_threshold:
                threshold_adapt = 0.9 * threshold_adapt + 0.1 * spikes
            
            # Update refractory period
            refractory = torch.where(spikes > 0.5, 
                                   torch.full_like(spikes, float(self.refractory_period)),
                                   torch.clamp(refractory - 1, min=0))
        
        # Update state dictionary
        updated_state_dict: StateDict = {
            'membrane': membrane,
            'synaptic': synaptic,
            'threshold_adapt': threshold_adapt,
            'refractory': refractory,
        }
        
        spike_train = torch.cat(spike_outputs, dim=1)
        
        if self.config.enable_logging and self.config.log_spike_stats and LOGURU_AVAILABLE:
            spike_rate = torch.mean(spike_train).item()
            logger.debug(f"LIF spike rate: {spike_rate:.4f}")
        
        return spike_train, updated_state_dict


class SpikingMambaLayer(nn.Module):
    """Spiking Mamba layer that integrates original Mamba with spiking dynamics"""
    
    def __init__(self, config: SpikingMambaConfig) -> None:
        super().__init__()
        self.config = config
        
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba package is required. Install with: pip install mamba-ssm")
        
        d_model = config.mamba_config.d_model
        
        # Core Mamba layer - use correct parameters
        mamba_kwargs = {
            'd_model': d_model,
            'd_state': config.mamba_config.d_state,
            'd_conv': config.mamba_config.d_conv,
            'expand': config.mamba_config.expand,
        }
        
        # Add optional parameters if they exist in config
        if hasattr(config.mamba_config, 'dt_rank') and config.mamba_config.dt_rank != "auto":
            mamba_kwargs['dt_rank'] = config.mamba_config.dt_rank
        if hasattr(config.mamba_config, 'conv_bias'):
            mamba_kwargs['conv_bias'] = config.mamba_config.conv_bias
        if hasattr(config.mamba_config, 'bias'):
            mamba_kwargs['bias'] = config.mamba_config.bias
        if hasattr(config.mamba_config, 'layer_idx'):
            mamba_kwargs['layer_idx'] = config.mamba_config.layer_idx
        
        # Choose Mamba version
        if config.use_mamba2 and MAMBA2_AVAILABLE:
            from mamba_ssm.modules.mamba2_simple import Mamba2
            self.mamba = Mamba2(**mamba_kwargs)
            if config.enable_logging and LOGURU_AVAILABLE:
                logger.info("Using Mamba2 architecture")
        else:
            self.mamba = Mamba(**mamba_kwargs)
            if config.enable_logging and LOGURU_AVAILABLE:
                logger.info("Using Mamba architecture")
        
        # Initialize spiking components based on integration mode
        self.pre_spiking: Optional[LIFNeuron] = None
        self.post_spiking: Optional[LIFNeuron] = None
        self.residual_spiking: Optional[LIFNeuron] = None
        
        if config.spike_integration in ["pre", "pre_post"]:
            self.pre_spiking = LIFNeuron(config, d_model)
        
        if config.spike_integration in ["post", "pre_post"]:
            self.post_spiking = LIFNeuron(config, d_model)
        
        if config.spike_integration == "residual":
            self.residual_spiking = LIFNeuron(config, d_model)
        
        # Spike-aware gating parameters
        if config.spike_gate_mode == "soft":
            self.spike_gate = nn.Parameter(torch.ones(d_model) * 0.5)
        
        # Temporal pooling mechanism
        if config.temporal_pooling == "learnable":
            self.temporal_pooling_weights = nn.Parameter(torch.ones(1) * 0.1)
        
        if config.enable_logging and LOGURU_AVAILABLE:
            logger.info(f"Initialized SpikingMambaLayer with integration mode: {config.spike_integration}")
    
    def forward(self, 
                x: TensorType, 
                inference_params: Optional[Any] = None, 
                spike_states: Optional[SpikeStates] = None) -> Tuple[TensorType, Optional[Any], SpikeStates]:
        """
        Forward pass of Spiking Mamba layer
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            inference_params: Mamba's inference parameters for efficient generation
            spike_states: Dictionary of spiking neuron states
            
        Returns:
            output: Processed tensor
            updated_inference_params: Updated inference parameters
            updated_spike_states: Updated spike states
        """
        batch_size, seq_len, d_model = x.shape
        
        if spike_states is None:
            spike_states = {}
        
        original_x = x
        
        # Pre-spiking processing
        if self.pre_spiking is not None:
            pre_spikes, spike_states['pre'] = self.pre_spiking(
                x, spike_states.get('pre', None)
            )
            
            if self.config.spike_gate_mode == "hard":
                x = x * pre_spikes  # Hard gating
            else:
                # Soft gating with learnable combination
                x = x * (self.spike_gate * pre_spikes + (1 - self.spike_gate))
        
        # Core Mamba processing
        if inference_params is not None:
            mamba_out = self.mamba(x, inference_params=inference_params)
        else:
            mamba_out = self.mamba(x)
        
        # Post-spiking processing
        if self.post_spiking is not None:
            post_spikes, spike_states['post'] = self.post_spiking(
                mamba_out, spike_states.get('post', None)
            )
            
            if self.config.spike_gate_mode == "hard":
                mamba_out = mamba_out * post_spikes
            else:
                mamba_out = mamba_out * (self.spike_gate * post_spikes + (1 - self.spike_gate))
        
        # Residual spiking connection
        if self.residual_spiking is not None:
            residual_spikes, spike_states['residual'] = self.residual_spiking(
                original_x, spike_states.get('residual', None)
            )
            mamba_out = mamba_out + self.spike_gate * residual_spikes
        
        # Temporal pooling
        if self.config.temporal_pooling == "adaptive":
            # Adaptive pooling based on spike activity
            spike_activity = sum(torch.mean(state_dict['membrane'].abs()).item() 
                               for state_dict in spike_states.values() 
                               if isinstance(state_dict, dict) and 'membrane' in state_dict)
            temporal_scale = torch.sigmoid(torch.tensor(spike_activity, device=x.device))
            mamba_out = mamba_out * temporal_scale
        elif self.config.temporal_pooling == "learnable":
            mamba_out = mamba_out * (1 + self.temporal_pooling_weights)
        
        return mamba_out, inference_params, spike_states


class SpikingMambaBlock(nn.Module):
    """Complete Spiking Mamba block with normalization and regularization"""
    
    def __init__(self, config: SpikingMambaConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        d_model = config.mamba_config.d_model
        
        # Set layer index for the mamba config
        config.mamba_config.layer_idx = layer_idx
        
        # RMS normalization (standard for Mamba)
        self.norm = nn.RMSNorm(d_model)
        
        self.spiking_mamba = SpikingMambaLayer(config)
        
        # Spike regularization strength
        self.spike_reg_strength = config.spike_regularization
    
    def forward(self, 
                x: TensorType, 
                inference_params: Optional[Any] = None, 
                spike_states: Optional[SpikeStates] = None) -> Tuple[TensorType, Optional[Any], SpikeStates, float]:
        """
        Forward pass of Spiking Mamba block
        
        Returns:
            output: Processed tensor with residual connection
            inference_params: Updated inference parameters
            spike_states: Updated spike states
            spike_reg_loss: Spike regularization loss
        """
        residual = x
        x = self.norm(x)
        x, inference_params, spike_states = self.spiking_mamba(x, inference_params, spike_states)
        
        # Add residual connection
        output = residual + x
        
        # Compute spike regularization loss
        spike_reg_loss = 0.0
        if self.training and self.spike_reg_strength > 0:
            for state_dict in spike_states.values():
                if isinstance(state_dict, dict) and 'membrane' in state_dict:
                    # L2 regularization on membrane potentials to encourage sparsity
                    spike_reg_loss += torch.mean(state_dict['membrane'] ** 2).item()
        
        return output, inference_params, spike_states, spike_reg_loss


@dataclass
class SpikeStats:
    """Data class for spike statistics"""
    total_spikes: int
    spike_rate: float
    layer_spike_rates: List[float]
    membrane_potentials: List[float]
    avg_membrane_potential: float
    max_membrane_potential: float
    
    def log_stats(self) -> None:
        """Log spike statistics if logging is enabled"""
        if LOGURU_AVAILABLE:
            logger.info("Spike Statistics:")
            logger.info(f"  Total spikes: {self.total_spikes}")
            logger.info(f"  Overall spike rate: {self.spike_rate:.4f}")
            logger.info(f"  Average membrane potential: {self.avg_membrane_potential:.4f}")
            logger.info(f"  Max membrane potential: {self.max_membrane_potential:.4f}")
            logger.info(f"  Layer spike rates: {[f'{rate:.4f}' for rate in self.layer_spike_rates]}")


@dataclass
class SpikingMambaOutput:
    """Data class for model output"""
    logits: TensorType
    inference_params_list: List[Optional[Any]]
    spike_states: List[SpikeStates]
    spike_reg_loss: float
    spike_stats: Optional[SpikeStats] = None


class SpikingMambaLM(nn.Module):
    """Complete Spiking Mamba Language Model with comprehensive typing and logging"""
    
    def __init__(self, config: SpikingMambaConfig) -> None:
        super().__init__()
        self.config = config
        
        if not MAMBA_AVAILABLE:
            raise ImportError("Mamba package is required. Install with: pip install mamba-ssm")
        
        mamba_config = config.mamba_config
        self.vocab_size = mamba_config.vocab_size
        self.d_model = mamba_config.d_model
        self.n_layers = mamba_config.n_layer
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Spiking Mamba layers
        self.layers = nn.ModuleList([
            SpikingMambaBlock(config, layer_idx=i) for i in range(self.n_layers)
        ])
        
        # Final normalization
        self.norm_f = nn.RMSNorm(self.d_model)
        
        # Output head
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        
        # Tie embeddings and output weights (common practice)
        self.lm_head.weight = self.embedding.weight
        
        # Initialize parameters
        self.apply(self._init_weights)
        
        if config.enable_logging and LOGURU_AVAILABLE:
            param_count = sum(p.numel() for p in self.parameters())
            logger.info(f"Initialized SpikingMambaLM with {param_count:,} parameters")
            logger.info(f"Architecture: {self.n_layers} layers, d_model={self.d_model}, vocab_size={self.vocab_size}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, 
                input_ids: TensorType, 
                inference_params_list: Optional[List[Optional[Any]]] = None, 
                spike_states_list: Optional[List[SpikeStates]] = None, 
                return_spike_stats: bool = False) -> SpikingMambaOutput:
        """
        Forward pass of Spiking Mamba Language Model
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            inference_params_list: List of inference parameters for each layer
            spike_states_list: List of spike states for each layer
            return_spike_stats: Whether to compute and return spike statistics
            
        Returns:
            SpikingMambaOutput containing logits, params, states, and optional statistics
        """
        batch_size, seq_len = input_ids.shape
        
        if inference_params_list is None:
            inference_params_list = [None] * self.n_layers
        if spike_states_list is None:
            spike_states_list = [None] * self.n_layers
        
        # Input validation
        if len(inference_params_list) != self.n_layers:
            raise ValueError(f"inference_params_list length ({len(inference_params_list)}) must match n_layers ({self.n_layers})")
        if len(spike_states_list) != self.n_layers:
            raise ValueError(f"spike_states_list length ({len(spike_states_list)}) must match n_layers ({self.n_layers})")
        
        # Embedding
        x = self.embedding(input_ids)
        
        if self.config.enable_logging and LOGURU_AVAILABLE:
            logger.debug(f"Processing input with shape: {input_ids.shape}")
        
        new_inference_params: List[Optional[Any]] = []
        new_spike_states: List[SpikeStates] = []
        total_spike_reg_loss = 0.0
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            x, inference_params, spike_states, spike_reg_loss = layer(
                x, inference_params_list[i], spike_states_list[i]
            )
            new_inference_params.append(inference_params)
            new_spike_states.append(spike_states)
            total_spike_reg_loss += spike_reg_loss
        
        # Final normalization and output
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        # Compute spike statistics if requested
        spike_stats = None
        if return_spike_stats:
            spike_stats = self._compute_spike_stats(new_spike_states)
            if self.config.enable_logging and self.config.log_spike_stats:
                spike_stats.log_stats()
        
        return SpikingMambaOutput(
            logits=logits,
            inference_params_list=new_inference_params,
            spike_states=new_spike_states,
            spike_reg_loss=total_spike_reg_loss,
            spike_stats=spike_stats
        )
    
    def _compute_spike_stats(self, spike_states_list: List[SpikeStates]) -> SpikeStats:
        """
        Compute comprehensive spiking statistics for analysis
        
        Args:
            spike_states_list: List of spike states from all layers
            
        Returns:
            SpikeStats object containing computed statistics
        """
        total_spikes = 0
        total_neurons = 0
        layer_spike_rates: List[float] = []
        membrane_potentials: List[float] = []
        
        for layer_states in spike_states_list:
            layer_spikes = 0
            layer_neurons = 0
            
            for component, state_dict in layer_states.items():
                if isinstance(state_dict, dict) and 'membrane' in state_dict:
                    membrane = state_dict['membrane']
                    # Approximate spike count from membrane dynamics
                    # This is a heuristic - actual spike counting would require storing spike trains
                    spikes_approx = torch.sum(membrane > self.config.threshold * 0.8).item()
                    layer_spikes += spikes_approx
                    layer_neurons += membrane.numel()
                    
                    membrane_potentials.append(torch.mean(membrane.abs()).item())
            
            if layer_neurons > 0:
                layer_spike_rate = layer_spikes / layer_neurons
                layer_spike_rates.append(layer_spike_rate)
                total_spikes += layer_spikes
                total_neurons += layer_neurons
        
        overall_spike_rate = total_spikes / total_neurons if total_neurons > 0 else 0.0
        avg_membrane_potential = sum(membrane_potentials) / len(membrane_potentials) if membrane_potentials else 0.0
        max_membrane_potential = max(membrane_potentials) if membrane_potentials else 0.0
        
        return SpikeStats(
            total_spikes=total_spikes,
            spike_rate=overall_spike_rate,
            layer_spike_rates=layer_spike_rates,
            membrane_potentials=membrane_potentials,
            avg_membrane_potential=avg_membrane_potential,
            max_membrane_potential=max_membrane_potential
        )


def create_spiking_mamba_model(
    d_model: int = 768,
    n_layer: int = 12,
    vocab_size: int = 50277,
    spike_mode: str = "pre_post",
    enable_logging: bool = True,
    use_mamba2: bool = False,
    **kwargs
) -> SpikingMambaLM:
    """
    Helper function to create a SpikingMamba model with proper configuration
    
    Args:
        d_model: Model dimension
        n_layer: Number of layers
        vocab_size: Vocabulary size
        spike_mode: Spiking integration mode
        enable_logging: Whether to enable logging
        use_mamba2: Whether to use Mamba2 architecture
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SpikingMambaLM model
    """
    
    # Base Mamba configuration
    mamba_config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        d_state=kwargs.get('d_state', 16),
        d_conv=kwargs.get('d_conv', 4),
        expand=kwargs.get('expand', 2),
    )
    
    # Spiking configuration with defaults
    spiking_config = SpikingMambaConfig(
        mamba_config=mamba_config,
        spike_integration=spike_mode,
        enable_logging=enable_logging,
        use_mamba2=use_mamba2,
        **{k: v for k, v in kwargs.items() if k not in ['d_state', 'd_conv', 'expand']}
    )
    
    if enable_logging and LOGURU_AVAILABLE:
        logger.info("Creating SpikingMamba model with configuration:")
        logger.info(f"  d_model={d_model}, n_layer={n_layer}, vocab_size={vocab_size}")
        logger.info(f"  spike_mode={spike_mode}, threshold={spiking_config.threshold}")
        logger.info(f"  use_mamba2={use_mamba2}")
    
    return SpikingMambaLM(spiking_config)


# Custom RMSNorm implementation for compatibility
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: TensorType) -> TensorType:
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


# Patch nn.RMSNorm if it doesn't exist (for older PyTorch versions)
if not hasattr(nn, 'RMSNorm'):
    nn.RMSNorm = RMSNorm


# Training utilities
class SpikingMambaTrainer:
    """Training utilities for SpikingMamba with spike-aware optimization"""
    
    def __init__(self, 
                 model: SpikingMambaLM, 
                 spike_loss_weight: float = 0.01,
                 enable_logging: bool = True):
        self.model = model
        self.spike_loss_weight = spike_loss_weight
        self.enable_logging = enable_logging
        
    def compute_loss(self, 
                     logits: TensorType, 
                     targets: TensorType, 
                     spike_reg_loss: float) -> Tuple[TensorType, Dict[str, float]]:
        """
        Compute total loss including spike regularization
        
        Args:
            logits: Model output logits
            targets: Target token IDs
            spike_reg_loss: Spike regularization loss
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Spike regularization loss
        spike_loss = torch.tensor(spike_reg_loss * self.spike_loss_weight, 
                                 device=logits.device, requires_grad=True)
        
        # Total loss
        total_loss = ce_loss + spike_loss
        
        loss_dict = {
            'cross_entropy': ce_loss.item(),
            'spike_regularization': spike_reg_loss,
            'total_loss': total_loss.item()
        }
        
        if self.enable_logging and LOGURU_AVAILABLE:
            logger.debug(f"Loss components: CE={ce_loss.item():.4f}, "
                        f"Spike={spike_reg_loss:.4f}, Total={total_loss.item():.4f}")
        
        return total_loss, loss_dict


# Inference utilities
class SpikingMambaGenerator:
    """Text generation utilities for SpikingMamba"""
    
    def __init__(self, 
                 model: SpikingMambaLM, 
                 tokenizer: Any = None,
                 enable_logging: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.enable_logging = enable_logging
        
    def generate(self, 
                 input_ids: TensorType,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 return_spike_stats: bool = False) -> Dict[str, Any]:
        """
        Generate text with SpikingMamba
        
        Args:
            input_ids: Starting token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            return_spike_stats: Whether to return spike statistics
            
        Returns:
            Dictionary containing generated tokens and optional spike stats
        """
        self.model.eval()
        batch_size, seq_len = input_ids.shape
        
        # Initialize states for generation
        inference_params_list = [None] * self.model.n_layers
        spike_states_list = [None] * self.model.n_layers
        
        generated_ids = input_ids.clone()
        all_spike_stats = [] if return_spike_stats else None
        
        with torch.no_grad():
            for _ in range(max_length - seq_len):
                # Forward pass
                output = self.model(
                    generated_ids,
                    inference_params_list=inference_params_list,
                    spike_states_list=spike_states_list,
                    return_spike_stats=return_spike_stats
                )
                
                # Update states for next iteration
                inference_params_list = output.inference_params_list
                spike_states_list = output.spike_states
                
                if return_spike_stats and output.spike_stats:
                    all_spike_stats.append(output.spike_stats)
                
                # Sample next token
                logits = output.logits[:, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Check for early stopping (e.g., EOS token)
                if self.tokenizer and hasattr(self.tokenizer, 'eos_token_id'):
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
        
        result = {
            'generated_ids': generated_ids,
            'new_tokens': generated_ids[:, seq_len:],
        }
        
        if return_spike_stats:
            result['spike_stats_history'] = all_spike_stats
            
        if self.enable_logging and LOGURU_AVAILABLE:
            logger.info(f"Generated {generated_ids.shape[1] - seq_len} tokens")
            
        return result
    
    