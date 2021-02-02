from typing import Optional, Tuple, Union, Type

import torch
import numpy as np

from ...losses_metrics import kld_normal, reconstruction_loss
from ...nets import fcEncoderNet, fcClassifier, init_VAE_nets
from ...trainers import viBaseTrainer
from ...utils import set_train_rng, to_onehot


class ssVAE(viBaseTrainer):

    def __init__(self,
                 in_dim: int = None,
                 latent_dim: int = 2,
                 nb_classes: int = 0,
                 aux_dim: int = 2,
                 seed: int = 0,
                 **kwargs: Union[int, bool, str]
                 ) -> None:
        """
        Initialize ssVAE module
        """
        super(ssVAE, self).__init__()
        in_dim_error_msg = (
            "You must specify the input dimensions and pass them as a tuple. "
            "For images, specify (height, width) or (height, width, channels)" +
            " if multiple channels. For spectra, specify (length,)")

        if in_dim is None or not isinstance(in_dim, (tuple, list)):
            raise AssertionError(in_dim_error_msg)
        if isinstance(in_dim, tuple) and not isinstance(in_dim[0], int):
            raise AssertionError(in_dim_error_msg)

        if torch.cuda.is_available:
            torch.cuda.empty_cache()
        set_train_rng(seed)
        np.random.seed(seed)

        self.in_dim = in_dim
        self.z_dim = latent_dim
        self.nb_classes = nb_classes
        self.aux_dim = aux_dim
        
        (encoder_net, decoder_net,
         self.metadict) = init_VAE_nets(
            in_dim, latent_dim, nb_classes=nb_classes,
            aux_dim=aux_dim, softplus_out=True, **kwargs)
        # Set main encoder and decoder
        self.set_model(encoder_net, decoder_net)
        # Set auxillary encoder and decoder
        self.set_aux_encoder(**kwargs)
        self.set_aux_decoder(**kwargs)
        # Set classifier
        self.set_classifier(**kwargs)
        # Add information about classifier atchitecture to dictionary

    def set_aux_encoder(self,
                        aux_encoder: Type[torch.nn.Module] = None,
                        **kwargs) -> None:
        if aux_encoder is None:
            numlayers = kwargs.get("numlayers_aux_encoder", 2)
            numhidden = kwargs.get("numhidden_aux_encoder", 128)
            aux_encoder = fcEncoderNet(
                self.in_dim, self.aux_dim, num_layers=numlayers,
                hidden_dim=numhidden, softplus_out=True)
            self.metadict["numlayers_aux_encoder"] = numlayers
            self.metadict["numhidden_aux_encoder"] = numhidden
        self.aux_encoder_net = aux_encoder

    def set_aux_decoder(self,
                        aux_decoder: Type[torch.nn.Module] = None,
                        **kwargs) -> None:
        if aux_decoder is None:
            in_dim = np.product(self.in_dim) + self.z_dim + self.nb_classes
            numlayers = kwargs.get("numlayers_aux_decoder", 2)
            numhidden = kwargs.get("numhidden_aux_decoder", 128)
            aux_decoder = fcEncoderNet(
                in_dim, self.aux_dim, num_layers=numlayers,
                hidden_dim=numhidden, softplus_out=True)
            self.metadict["numlayers_aux_decoder"] = numlayers
            self.metadict["numhidden_aux_decoder"] = numhidden
        self.aux_decoder_net = aux_decoder

    def set_classifier(self,
                       cls_net: Type[torch.nn.Module] = None,
                       **kwargs) -> None:
        if cls_net is None:
            numhidden = kwargs.get("numhidden_cls", 128)
            numlayers = kwargs.get("numlayers_cls", 1)
            cls_net = fcClassifier(
                self.in_dim, self.nb_classes,
                hidden_dim=numhidden, num_layers=numlayers)
            self.metadict["numlayers_cls"] = numlayers
            self.metadict["numhidden_cls"] = numhidden
        self.cls_net = cls_net

    def encoders(self, x, y):
        """
        Main & auxillary encoders
        """
        # Auxiliary inference q(a|x)
        q_a_mean, q_a_logsd = self.aux_encoder_net(x)
        # reparametrization
        q_a_sd = torch.exp(q_a_logsd)
        q_a = self.reparameterize(q_a_mean, q_a_sd)

        # Latent inference q(z|a,y,x)
        x = x.view(-1, np.product(x.shape[1:]))
        z_mean, z_logsd = self.encoder_net(torch.cat([x, y, q_a], dim=1))
        # reparametrization
        z_sd = torch.exp(z_logsd)
        z = self.reparameterize(z_mean, z_sd)

        return (q_a_mean, q_a_logsd), (z, z_mean, z_logsd)

    def decoders(self, z, y, x):
        """
        Main & auxillary decoders
        """
        # Main generative process p(x|z,y)
        z_vec = torch.cat((z, y), -1)
        x_reconstr = self.decoder_net(z_vec)

        # auxillary generative p(a|z,y,x)
        x = x.view(-1, np.product(x.shape[1:]))
        z_vec = torch.cat((x, y, z), -1)
        p_a_mean, p_a_log_sd = self.aux_decoder_net(z_vec)

        return x_reconstr, (p_a_mean, p_a_log_sd)

    def elbo_fn(self, x, x_reconstr, *args):
        """
        Computes ELBO
        """
        likelihood = -reconstruction_loss(
            self.loss, self.in_dim, x, x_reconstr, logits=True).sum(-1)
        z_mean, z_log_sd, q_a_mean, q_a_logsd, p_a_mean, p_a_logsd = args
        kl_main = kld_normal([z_mean, z_log_sd])
        kl_aux = kld_normal([q_a_mean, q_a_logsd], [p_a_mean, p_a_logsd])
        return likelihood - (kl_main + kl_aux)

    def sample_labeled(self, mode: str = "train") -> Tuple[torch.Tensor]:
        bsize = self.batch_size
        if mode == "train":
            (X, y) = (self.X_train_l, self.y_train_l)
        else:
            (X, y) = (self.X_test_l, self.y_test_l)
        num_batches = len(X) // bsize
        bidx = np.random.randint(num_batches)
        X_sampled = X[bidx * bsize: (bidx + 1) * bsize]
        y_sampled = y[bidx * bsize: (bidx + 1) * bsize]
        y_sampled = to_onehot(y_sampled, self.nb_classes)
        return X_sampled, y_sampled
            
    def _forward_compute_elbo(self, x, y=None):
        """
        """
        if y is None:
            ys = self.enumerate_discrete(x)
            xs = x.repeat_interleave(self.nb_classes, 0)
        else:
            xs, ys = x, y
        # Get output of all (main+auxillary) encoders and decoders
        (q_a_mean, q_a_logsd), (z, z_mean, z_logsd) = self.encoders(xs, ys)
        x_reconstr, (p_a_mean, p_a_log_sd) = self.decoders(z, ys, xs)
        # Compute ELBO
        args = (z_mean, z_logsd, q_a_mean, q_a_logsd, p_a_mean, p_a_log_sd)
        return self.elbo_fn(xs, x_reconstr, *args)

    def forward_compute_elbo(self, x, mode="train"): # add train and eval modes
        # Compute ELBO for unlabeled data
        L = self._forward_compute_elbo(x)
        # Sample from labeled data
        x_l, y_l = self.sample_labeled(mode)
        # Compute ELBO for the sampled labeled data
        L_m = self._forward_compute_elbo(x_l, y_l).mean()
        # Add classification loss on unlabeled part
        logits_softmax = self.cls_net(x)
        L = L.view_as(logits_softmax.t()).t()
        H = -torch.sum(logits_softmax * torch.log(logits_softmax + 1e-8), dim=-1)
        L = torch.sum(logits_softmax * L, -1)
        U_m = torch.mean(L + H)  # -U(x) in Eq(11)
        # Add classification loss on labeled part
        logits_softmax = self.cls_net(x_l)
        cls_loss = -torch.sum(y_l * torch.log(logits_softmax + 1e-8), dim=-1)

        J = -L_m - U_m + self.alpha * cls_loss.mean()  # Eq(13)

        return J

    def fit(self,
            X_train: Union[np.ndarray, torch.Tensor],
            y_train: Union[np.ndarray, torch.Tensor],
            X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
            loss: str = "ce",
            **kwargs) -> None:
        """
        """
        X_train_l, y_train_l, X_train_u = self.split_data(X_train, y_train)
        X_train_l, y_train_l = self._2torch(X_train_l, y_train_l)
        if X_test is not None and y_test is not None:
            X_test_l, y_test_l, X_test_u = self.split_data(X_test, y_test)
            X_test_l, y_test_l = self._2torch(X_test_l, y_test_l)
        else:
            X_test_l = y_test_l = X_test_u = None
        self.compile_trainer(
            (X_train_u, None), (X_test_u, None), **kwargs)
        self.alpha = 0.1 * (len(X_train_u) + len(X_train_l)) / len(X_train_l)
        self.X_train_l, self.y_train_l = X_train_l, y_train_l
        self.X_test_l, self.y_test_l = X_test_l, y_test_l
        self.loss = loss

        for e in range(self.training_cycles):
            self.current_epoch = e
            elbo_epoch = self.train_epoch()
            self.loss_history["train_loss"].append(elbo_epoch)
            if self.test_iterator is not None:
                elbo_epoch_test = self.evaluate_model()
                self.loss_history["test_loss"].append(elbo_epoch_test)
            print(e)
            # self.print_statistics(e)  need to define for classification loss
            self.save_model(self.filename)

    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   ) -> Tuple[np.ndarray]:
        """
        Splits data into labelled and unlabelled sets
        """
        if not np.isnan(y).any():
            raise ValueError("Missing labels must be provoded as NaNs")
        X_labeled = X[~np.isnan(y)]
        X_unlabeled = X[np.isnan(y)]
        y_labeled = y[~np.isnan(y)]
        return X_labeled, y_labeled, X_unlabeled

    def enumerate_discrete(self, x):
        
        def batch(batch_size, label):
            labels = (torch.ones(batch_size, 1) * label).type(torch.LongTensor)
            y = torch.zeros((batch_size, self.nb_classes))
            y.scatter_(1, labels, 1)
            return y

        batch_size = x.size(0)
        generated = torch.cat(
            [batch(batch_size, i) for i in range(self.nb_classes)])

        return generated.to(self.device)

    def _2torch(self,
                X: Union[np.ndarray, torch.Tensor],
                y: Union[np.ndarray, torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Rules for conversion of numpy arrays to torch tensors
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        return X, y
