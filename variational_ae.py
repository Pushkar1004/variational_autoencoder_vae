import torch
import torch.nn as nn
import torch.nn.functional as F

"""
N is batch size
2500 is the size of the input signal
z_dim is the dimension of the latent space
number of channels in input array is 1. 
so input size becomes (N,1,2500)
"""
        
# Encoder 
class Encoder1D(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)    # (N, 1,2500)-> (N, 32, 1250)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)   # (N, 32, 1250)-> (N, 64, 625)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)  # (N, 64, 625) -> (N, 128, 312)
        self.conv4 = nn.Conv1d(128,256, kernel_size = 4, stride = 2, padding=1) # (N, 128, 312) -> (N, 256,156)
        self._flatten_dim = 256 * 156 # _flatten_dim: dimension of the bottleneck
        self.fc_mu = nn.Linear(self._flatten_dim, z_dim)    #Linear layer that returns mean
        self.fc_logvar = nn.Linear(self._flatten_dim, z_dim) #Linear Layer that returns log(variance)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)   # change x from (N , 256, 156) -> (N, 256*156) to create bottleneck
        mu = self.fc_mu(x)          # returns (N,z_dim) for mean
        logvar = self.fc_logvar(x)   # returns (N, z_dim) for variance
        return mu, logvar

# Decoder
class Decoder1D(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256 * 156)   # (N, Z_dim) -> (N, 256*156) 
        self.deconv1 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1) #(N,256*156) -> (N,128,312)
        self.deconv2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1) # (N, 128,312) -> (N, 64, 625)
        self.deconv3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1) #(N,64,625) -> (N,32,1250)
        self.deconv4 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1) #(N, 32,1250) -> (N,1,2500)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 156)  #Change x from (N, 256*156) -> (N, 256,156) 
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = (self.deconv4(x))  
        return x

class VAE1D(nn.Module):
    def __init__(self, z_dim = 20):
        super().__init__()
        self.encoder = Encoder1D(z_dim)
        self.decoder = Decoder1D(z_dim)
    
    def reparameterize(self, mu, logvar):
        # Directly sampling from mu and logvar blocks gradient flow, breaking gradient-based optimization.
        # The reparameterization trick allows gradients to flow from the decoder to the encoder during training.
        # Process:
            #  1. Sample epsilon ~ N(0, 1)
            #  2. Scale by standard deviation (sigma) derived from logvar
            #  3. Shift by mean (mu) to get the final latent vector
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar  # return all three as they are used in loss function 

def vae_loss_function(recon_x, x, mu, logvar, beta = 1.0):   
    """
    Parameters
    recon_x : torch.Tensor, The reconstructed output from the decoder.
    x : torch.Tensor, The original input data.
    mu : torch.Tensor, Mean of the latent distribution (encoder output).
    logvar : torch.Tensor, Log-variance of the latent distribution (encoder output).
    beta : float, optional
           Weight for the KL divergence term (useful for β-VAE).
           Default is 1.0 (standard VAE).

    Returns
    total_loss : torch.Tensor, The sum of reconstruction loss and KL divergence.
    recon_loss : torch.Tensor, The reconstruction (MSE) loss.
    kl_loss : torch.Tensor, The KL divergence.
    """
    # Reconstruction loss (Mean Squared Error)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    # KL divergence (closed-form between q(z|x) ~ N(mu, sigma^2) and p(z) ~ N(0, I))    
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
        