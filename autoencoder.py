import torch.nn as nn
import torch.nn.functional as F

# Encoder 
class Encoder1D(nn.Module):
    def __init__(self, z_dim=20):
        super().__init__()
        
        """
        N is batch size
        2500 is the size of the signal that we want to encode
        Z is the dimension of the latent space
        number of channels in array is 1. 
        so input size becomes (N,1,2500)
        """
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1)    # (N, 1,2500)-> (N, 32, 1250)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)   # (N, 32, 1250)-> (N, 64, 625)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)  # (N, 64, 625) -> (N, 128, 312)
        self.conv4 = nn.Conv1d(128,256, kernel_size = 4, stride = 2, padding=1) # (N, 128, 312) -> (N, 256,156)
        
        self._flatten_dim = 256 * 156 # _flatten_dim: dimension of the bottleneck
        self.latent = nn.Linear(self._flatten_dim, z_dim) # (N, 256*156) -> (N, Z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)         # change x from (N , 256, 156) -> (N, 256*156) to create bottleneck
        z = self.latent(x)
        return z

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
        x = ((self.deconv4(x))) 
        return x

class autoencoder1D(nn.Module):
    def __init__(self, z_dim = 20):
        super().__init__()
        self.encoder = Encoder1D(z_dim)
        self.decoder = Decoder1D(z_dim)
    
    def forward(self, x):
        z = self.encoder(x) 
        recon = self.decoder(z) 
        return recon

# loss = F.mse_loss(recon, x)

