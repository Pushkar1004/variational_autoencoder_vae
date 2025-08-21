import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Input format:
  N -> Batch size (instances from a single group, since we want to accumulate information from events of the same group).
  Signal length -> 2500 (1D signal to encode).
  Channels -> 1.
  Latent space dimension -> Z.

Thus, input shape = (N, 1, 2500).
"""

# Encoder for Coherent Information
class EncoderCoh1D(nn.Module):
    """
    Encodes coherent information from each instance.
    This information is shared across all events in the group.
    Precision-based accumulation of the outputs (mu, logvar) is done later.
    """
    def __init__(self, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)    # (N, 1,2500)-> (N, 32, 1250)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)   # (N, 32, 1250)-> (N, 64, 625)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)  # (N, 64, 625) -> (N, 128, 312)
        self.conv4 = nn.Conv1d(128,256, kernel_size = 4, stride = 2, padding=1) # (N, 128, 312) -> (N, 256,156)
        self._flatten_dim = 256 * 156 # _flatten_dim: dimension of the bottleneck
        self.fc_mu = nn.Linear(self._flatten_dim, z_dim) # returns mean of coherent latent distribution
        self.fc_logvar = nn.Linear(self._flatten_dim, z_dim) # returns log-variance of coherent latent distribution


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten (N, 256, 156) -> (N, 256*156)
        mu_coh = self.fc_mu(x)         # (N,z_dim) 
        logvar_coh = self.fc_logvar(x) # (N, z_dim) 
        # logvar = torch.clamp(logvar, min=-10, max=10)
        return mu_coh, logvar_coh

# Encoder for Nuisance Information
class EncoderNui1D(nn.Module):
    """
    Encodes nuisance information for each instance.
    Unlike coherent information, nuisance information is unique to each event in the group.
    """
    def __init__(self, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        # Input shape: (N, 1, 2500)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)    # (N, 1,2500)-> (N, 32, 1250)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)   # (N, 32, 1250)-> (N, 64, 625)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)  # (N, 64, 625) -> (N, 128, 312)
        self.conv4 = nn.Conv1d(128,256, kernel_size = 4, stride = 2, padding=1) # (N, 128, 312) -> (N, 256,156)
        self._flatten_dim = 256 * 156 # _flatten_dim: dimension of the bottleneck
        self.fc_mu = nn.Linear(self._flatten_dim, z_dim) # returns mean of nuisance latent distribution
        self.fc_logvar = nn.Linear(self._flatten_dim, z_dim) # returns log-variance of nuisance latent distribution

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # flatten (N, 256, 156) -> (N, 256*156)
        mu_nui = self.fc_mu(x)         # (N, z_dim)
        logvar_nui = self.fc_logvar(x) # (N, z_dim)
        # logvar = torch.clamp(logvar, min=-10, max=10)
        return mu_nui, logvar_nui


# Decoder
class Decoder1D(nn.Module):
    """
    Reconstructs the original signal from the latent representation.
    Input:
      Latent vector formed by concatenating coherent (shared) and nuisance (instance-specific) latent variables.
      Decoder input dimension = z_dim_coherent + z_dim_nuisance
    """
    def __init__(self, z_dim=40):
        super().__init__()
        self.z_dim = z_dim
        self.fc = nn.Linear(z_dim, 256 * 156)  # (N, z_dim) -> (N, 256*156) 
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)  #(N,256*156) -> (N,128,312)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)  # (N, 128,312) -> (N, 64, 625)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) #(N,64,625) -> (N,32,1250)
        self.deconv4 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)  #(N, 32,1250) -> (N,1,2500)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 156)  # reshape (N, 256*156) -> (N, 256, 156)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = (self.deconv4(x)) 
        return x

        

class SYMVAE1D(nn.Module):
    """
    Symmetric Variational Autoencoder:
      Splits latent space into coherent (shared across group) and nuisance (instance-specific).
      Performs precision-based accumulation of coherent latent variables.
      Reconstructs input from concatenated latent vectors.
    """
    def __init__(self, z_dim_coh=20, z_dim_nui = 20):
        super().__init__()
        self.encoderCoh = EncoderCoh1D(z_dim_coh)
        self.encoderNui = EncoderNui1D(z_dim_nui)
        self.decoder = Decoder1D(self.encoderCoh.z_dim + self.encoderNui.z_dim)

    def reparameterize(self, mu, logvar):
        """
        Directly sampling from mu and logvar blocks gradient flow, breaking gradient-based optimization.
        The reparameterization trick allows gradients to flow from the decoder to the encoder during training.
            
            Sample epsilon ~ N(0, 1)
            Scale by standard deviation (sigma) derived from logvar
            Shift by mean (mu) to get the final latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def accumulate_Gaussians(self, mu, logvar):
        """
        Accumulates Gaussian distributions across a group of instances. 
        Uses precision-weighted averaging of means.

        Parameters
            mu -> torch.Tensor
                Tensor of shape (N, z_dim) representing the means of N instances in the group.
            logvar -> torch.Tensor
                Tensor of shape (N, z_dim) representing the log(variances) of N instances in the group.

        Returns
            c_mu -> torch.Tensor
                Accumulated mean (1, z_dim) obtained via precision-weighted averaging.
            c_logvar -> torch.Tensor
                Accumulated log-variance (1, z_dim) computed from the summed precision.

        Notes
            Precision is defined as the inverse of variance-> precision = 1 / var
            Low variance (high precision) -> distribution is more informative 
                (tighter around its mean).
            High variance (low precision) -> distribution is less informative 
                (broader spread).
            Accumulation is done by summing precision, not variance:
                Variances are not directly summed (since they only represent spread).
                Precisions are summed, since they quantify informativeness.
                The accumulated mean is computed as the weighted mean with precision weights.
                The accumulated variance is `1 / (sum of precisions)`.
        """
    
        var = torch.exp(logvar)
        precision = 1 / var
        c_pre = precision.sum(dim=0, keepdim=True)         # accumulated precision
        c_var = 1 / c_pre                                  # accumulated variance
        muXpre = torch.sum(mu * precision, dim=0, keepdim=True)
        c_mu = muXpre / c_pre                              # accumulated mean
        c_logvar = torch.log(c_var)
        return c_mu, c_logvar


    def forward(self, x):
        # Encode
        mu_coh, logvar_coh = self.encoderCoh(x)
        mu_nui, logvar_nui = self.encoderNui(x)
        # Accumulate coherent latent variables across group
        c_mu_coh, c_logvar_coh = self.accumulate_Gaussians(mu_coh, logvar_coh)
        # Broadcast shared coherent latent vars across all events in group
        c_mu_expanded = c_mu_coh.expand(x.shape[0], 20)  
        c_logvar_expanded = c_logvar_coh.expand(x.shape[0], 20) 
        # Concatenate nuisance (instance-specific) with coherent (shared) latent vars
        mu = torch.cat((mu_nui, c_mu_expanded), dim=1)
        logvar = torch.cat((logvar_nui, c_logvar_expanded), dim=1)
        # Reparameterize and decode
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta = 1.0):   
    """
    Parameters
    recon_x -> torch.Tensor
        The reconstructed output from the decoder.
    x -> torch.Tensor 
        The original input data.
    mu -> torch.Tensor 
        Mean of the latent distribution (concatenated from nuisance and accumulated coherent means).
    logvar -> torch.Tensor 
        Log-variance of the latent distribution (concatenated from nuisance and accumulated coherent log-variances).
    beta -> float, optional
        Weight for the KL divergence term (useful for β-VAE).
        Default is 1.0 (standard VAE).

    Returns
        total_loss -> torch.Tensor, The sum of reconstruction loss and KL divergence.
        recon_loss -> torch.Tensor, The reconstruction (MSE) loss.
        kl_loss -> torch.Tensor, The KL divergence.
    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    # KL divergence (closed-form between q(z|x) ~ N(mu, sigma^2) and p(z) ~ N(0, I))    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss


