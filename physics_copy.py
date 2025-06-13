"""
Physics module for MIWEN implementation containing physical constants,
noise models, and mixing functions.
"""

import torch
import scipy.constants as const
from dataclasses import dataclass

@dataclass
class PhysicalConstants:
    """Physical constants and device parameters"""
    kb: float = const.k        # Boltzmann constant
    T: float = 300.0          # Room temperature (K)
    e: float = const.e        # Elementary charge
    c: float = const.c        # Speed of light
    h: float = const.h        # Planck constant
    hbar: float = const.hbar  # Reduced Planck constant
    R: float = 50.0          # Load resistance (Ω)
    
    @property
    def VT(self):
        """Thermal voltage"""
        return self.kb * self.T / self.e

class NoiseModel:
    """Implements various noise models for the system"""
    def __init__(self, constants: PhysicalConstants, bandwidth: float):
        self.constants = constants
        self.bandwidth = bandwidth
    
    def thermal_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Add Johnson-Nyquist noise to the signal"""
        # Thermal noise power = 4kTR∆f
        noise_power = 4 * self.constants.kb * self.constants.T * self.constants.R * self.bandwidth
        noise_std = torch.sqrt(torch.tensor(noise_power))
        noise = torch.randn_like(signal) * noise_std
        return signal + noise

    def shot_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Add shot noise (optional)"""
        # TODO: Implement shot noise
        return signal

class DiodeMixing:
    """Implements different diode mixing models"""
    def __init__(self, constants: PhysicalConstants, noise_model: NoiseModel):
        self.constants = constants
        self.noise_model = noise_model

    def simple_mixing(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Simplified mixing with tanh for stability"""
        z_noisy = self.noise_model.thermal_noise(z)
        w_noisy = self.noise_model.thermal_noise(w)
        return torch.tanh((z_noisy + w_noisy) / (2 * self.constants.VT))

    def exact_mixing(self, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        EXACT diode formula with numerical stability:
        out = w/2 + (VT/2)* log( (exp(z/VT)+ exp(-w/VT)) / (exp(z/VT)+ exp(w/VT)) )
        """
        z_noisy = self.noise_model.thermal_noise(z)
        w_noisy = self.noise_model.thermal_noise(w)
        
        VT = self.constants.VT
        eps = 1e-8
        
        z_scaled = torch.clamp(z_noisy / VT, min=-50, max=50)
        w_scaled = torch.clamp(w_noisy / VT, min=-50, max=50)
        
        max_val = torch.maximum(z_scaled, torch.maximum(w_scaled, -w_scaled))
        numerator = torch.log(torch.exp(z_scaled - max_val) + torch.exp(-w_scaled - max_val) + eps) + max_val
        denominator = torch.log(torch.exp(z_scaled - max_val) + torch.exp(w_scaled - max_val) + eps) + max_val
        
        out = (w_noisy/2.0) + (VT/2.0) * (numerator - denominator)
        out_noisy = self.noise_model.thermal_noise(out)
        return torch.clamp(out_noisy, min=-1e3, max=1e3) 
        