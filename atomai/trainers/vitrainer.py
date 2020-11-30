

class viBaseTrainer:

    def __init__():
        # itialize empty arguments, etc.
        self.z_dim = 1
        self.encoder_net = None
        self.decoder_net = None
        self.train_iterator = None
        self.test_iterator = None
        self.optim = None

    def set_model(encoder_net, decoder_net):
        self.encoder_net = encoder_net
        self.decoder_net = decoder_net

    def set_data(X_train, X_test):
        #self.train_iterator = 
        #self.test_iterator = 
        pass

    def step(self, x: torch.Tensor, mode: str = "train") -> None: # simple vae step here
        pass

    def compile_trainer(self): # compile train.test iterators and optimizer
        pass

    @classmethod
    def reparameterize(cls, z_mean: torch.Tensor,
                       z_sd: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick
        """
        batch_dim = z_mean.size(0)
        z_dim = z_mean.size(1)
        eps = z_mean.new(batch_dim, z_dim).normal_()
        return z_mean + z_sd * eps

    def train_epoch(self):
        """
        Trains a single epoch
        """
        self.decoder_net.train()
        self.encoder_net.train()
        c = 0
        elbo_epoch = 0
        for x in self.train_iterator:
            if len(x) == 1:
                x = x[0]
                y = None
            else:
                x, y = x
            b = x.size(0)
            elbo = self.step(x) if y is None else self.step(x, y)
            loss = -elbo
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch)
            elbo_epoch += delta / c
        return elbo_epoch

    def evaluate_model(self):
        """
        Evaluates model on test data
        """
        self.decoder_net.eval()
        self.encoder_net.eval()
        c = 0
        elbo_epoch_test = 0
        for x in self.test_iterator:
            if len(x) == 1:
                x = x[0]
                y = None
            else:
                x, y = x
            b = x.size(0)
            if y is None:
                elbo = self.step(x, mode="eval")
            else:
                elbo = self.step(x, y, mode="eval")
            elbo = elbo.item()
            c += b
            delta = b * (elbo - elbo_epoch_test)
            elbo_epoch_test += delta / c
        return elbo_epoch_test
