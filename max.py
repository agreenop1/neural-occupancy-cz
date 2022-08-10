# Neural Dynamic Occupancy Models for Breeding Bird Survey Data

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import itertools
import re
import multiprocessing

import torch

torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import os
import random

# custom modules
import dataset
import utils

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

b1=pd.read_csv("data/cleaned/bbs.csv")
b2=pd.read_csv("data/cleaned/clean_routes.csv")
# Load all of the BBS data into a DataFrame
bbs = pd.merge(b1,
    b2,
    how="left",
    on="route_id",
)

bbs["genus"].unique # get unique categories in genus

# Create dictionaries that map categories to integers
categorical_covariates = ["english", "genus", "family", "order", "L1_KEY"]
cat_ix = {}
for cat in categorical_covariates:
    cat_dict = {}
    category_series = bbs[cat].unique()
    for i, item in enumerate(category_series):
        cat_dict[item] = i
    cat_ix[cat] = cat_dict


class BBSData(Dataset):
    """North American Breeding Bird Survey data."""

    def __init__(self, df):
        """
        Args:
            df (pandas.DataFrame): data frame with bbs data.
        """
        self.df = df
        self.cat_ix = cat_ix
        self.bbs_y = self.get_cont("^[0-9]{4}$", df)
        self.x_p = torch.stack(
            (
                self.get_cont("^StartTemp_", df),
                self.get_cont("^StartWind_", df),
                self.get_cont("^EndTemp_", df),
                self.get_cont("^EndWind_", df),
                self.get_cont("^StartSky_", df),
                self.get_cont("^EndSky_", df),
                self.get_cont("^duration_", df),
            ),
            -1,
        )
        self.bbs_species = self.get_cat("english", df)
        self.bbs_genus = self.get_cat("genus", df)
        self.bbs_family = self.get_cat("family", df)
        self.bbs_order = self.get_cat("order", df)
        self.bbs_l1 = self.get_cat("L2_KEY", df)
        # x is a covariate vector, e.g., PC1, PC2, ...
        self.bbs_x = torch.tensor(
            df.filter(regex="^PC|^c_", axis=1).values, dtype=torch.float64
        )

    def __len__(self):
        return len(self.df.index)

    def get_cont(self, regex, df):
        """ Extract continuous valued data from columns matching regex. """
        res = torch.tensor(
            df.filter(regex=regex, axis=1).values, dtype=torch.float64
        )
        return res

    def get_cat(self, name, df):
        """ Find an integer index for a particular category. """
        res = [self.cat_ix[name][i[0]] for i in self.df[[name]].values]
        res = np.array(res, dtype=np.long)
        return torch.tensor(res, dtype=torch.long)

    def __getitem__(self, idx):
        """ Get an item from the data.

        This is one training example: a route X species time series with feats
        """
        y = self.bbs_y[idx, :].squeeze(0)
        x_p = self.x_p[idx, :, :].squeeze(0)
        x = self.bbs_x[idx, :].squeeze(0)
        species = self.bbs_species[idx]
        genus = self.bbs_genus[idx]
        family = self.bbs_family[idx]
        order = self.bbs_order[idx]
        l1 = self.bbs_l1[idx]
        return species, genus, family, order, l1, x, x_p, y
# Set device to the gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cudnn.deterministic = True
cudnn.benchmark = True


# Set seed for reproducibility
def set_seed(seed):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(2147483647)

## Reading and parsing the Breeding Bird Survey Data

bbs_train = dataset.BBSData(dataset.bbs.query("group == 'train'"))
bbs_valid = dataset.BBSData(dataset.bbs.query("group == 'validation'"))

# loading one example
ex_sp, ex_gn, ex_fm, ex_or, ex_l1, ex_x, ex_x_p, ex_y = bbs_train[0]

# checking shapes
nx = ex_x.shape[0]
nt = ex_y.shape[0]
nsp = len(dataset.cat_ix["english"].keys())
ngn = len(dataset.cat_ix["genus"].keys())
nfm = len(dataset.cat_ix["family"].keys())
nor = len(dataset.cat_ix["order"].keys())
nx_p = ex_x_p.shape[-1]

print(f"Dimension of x is {nx}")
print(f"Number of time steps is {nt}")
print(f"Number of species is {nsp}")
print(f"Number of genera is {ngn}")
print(f"Number of families is {nfm}")
print(f"Number of orders is {nor}")
print(f"Dimension of detection covariate vector is {nx_p}")

##########################################################################
##########################################################################
# Training species-specific models
class OneSpeciesNet(nn.Module):
    """ Single species neural dynamic occupancy model. """

    embed_dim = 2 ** 3
    h_dim = 2 ** 2
    activation = nn.LeakyReLU()

    def __init__(self, num_l1, nx, nt, nx_p):
        """Initialize a network object.

        Args
        ----
        num_l1 (int): number of level 1 ecoregions
        nx (int): number of route-level input features
        nt (int): number of timesteps (years)
        nx_p (int): number of detection/survey-level covariates
        """

        # Define embeddings and fully connected layers
        super(OneSpeciesNet, self).__init__()
        self.nt = nt
        self.nx_p = nx_p

        # shared layers
        self.l1_embed = nn.Embedding(num_l1, self.embed_dim)

        self.to_h = nn.Sequential(
            nn.Linear(self.embed_dim + nx, self.h_dim), self.activation
        )

        # component output layers
        self.to_psi0 = nn.Linear(self.h_dim, 1)
        self.to_p = nn.Linear(self.h_dim + nx_p, 1)
        self.to_phi = nn.Linear(self.h_dim, 1)
        self.to_gamma = nn.Linear(self.h_dim, 1)

    def forward(self, sp, gn, fm, order, l1, x, x_p):
        """ Make a forward pass for the model.
        Note that while species, genus, family, and order are provided as
        inputs, these are ignored. (Keeping them allows the re-use of the
        same dataset definition.)

        Args
        ----
        sp (tensor): integer indices for species
        gn (tensor): integer indices for genera
        fm (tensor): integer indices for families
        order (tensor): integer indices for orders
        l1 (tensor): integer indices for level 1 ecoregion
        x (tensor): route-level features
        x_p (tensor): survey-level features

        Return
        ------
        A dictionary of parameters
        """

        l1_vec = self.l1_embed(l1)
        h = self.to_h(torch.cat((l1_vec, x), -1))

        phi = torch.sigmoid(self.to_phi(h)).repeat(1, self.nt - 1)
        gamma = torch.sigmoid(self.to_gamma(h)).repeat(1, self.nt - 1)
        psi0 = torch.sigmoid(self.to_psi0(h))

        # for each year, concatenate the route embedding with survey condtion data
        p_t = list()
        for t in range(self.nt):
            # stack hidden state with survey data
            p_t.append(self.to_p(torch.cat((h, x_p[:, t, :]), -1)))
        logit_p = torch.stack(p_t, dim=-1).squeeze()  # batch_size X nt
        out_dict = {
            "phi": phi,
            "gamma": gamma,
            "psi0": psi0,
            "logit_p": logit_p,
        }
        return out_dict


def fit_single_species_model(sp, model):
    """ Fit a single species model.

    This function trains a single species model and writes the
    results to a csv file.

    Args:
    - sp (string): english species name
    - model (nn.Module): the model to use
    """
    sp_name_lower = re.sub(" ", "_", sp.lower())
    out_path = f"out/{sp_name_lower}_ssnet.csv"
    if os.path.isfile(out_path):
        return f"File already exists for {sp}. Moving on..."

    sp_train = dataset.BBSData(
        dataset.bbs.query("group == 'train' & english == @sp")
    )
    sp_valid = dataset.BBSData(
        dataset.bbs.query("group == 'validation' & english == @sp")
    )
    sp_train_loader = DataLoader(
        sp_train, batch_size=batch_size // 4, shuffle=True, num_workers=multiprocessing.cpu_count()
    )
    sp_valid_loader = DataLoader(
        sp_valid, batch_size=batch_size * 4, num_workers=multiprocessing.cpu_count(), shuffle=True
    )
    sp_net = model(
        num_l1=len(dataset.cat_ix["L1_KEY"]), nx=nx, nt=nt, nx_p=nx_p
    )
    sp_net.to(device)

    optimizer = optim.Adam(sp_net.parameters(), weight_decay=1e-3, lr=0.01)
    max_n_epoch = 100

    train_loss = list()
    valid_loss = list()
    for epoch in range(max_n_epoch):
        with torch.enable_grad():
            sp_net.train()
            train_loss.append(
                utils.fit_epoch(
                    sp_net,
                    sp_train_loader,
                    training=True,
                    optimizer=optimizer,
                    pb=False,
                )
            )
        with torch.no_grad():
            sp_net.eval()
            valid_loss.append(
                utils.fit_epoch(
                    sp_net, sp_valid_loader, training=False, pb=False
                )
            )
            if epoch > 2:
                valid_loss_diff = np.mean(valid_loss[-1]) - np.mean(
                    valid_loss[-2]
                )
                if valid_loss_diff >= -0.2:
                    break

    valid_arr = np.concatenate(valid_loss)
    train_arr = np.concatenate(train_loss)

    sp_df = dataset.bbs.query("english == @sp & group != 'test'")
    years = list(sp_df.filter(regex="^[0-9]{4}", axis=1))
    sp_ds = dataset.BBSData(sp_df)
    sp_loader = DataLoader(
        sp_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=multiprocessing.cpu_count(),
        pin_memory=True,
    )
    sp_p = []
    sp_gamma = []
    sp_phi = []
    sp_psi0 = []
    for i_batch, xy in enumerate(sp_loader):
        _, out = utils.bbs_nll(xy, model=sp_net)
        sp_p.append(torch.sigmoid(out["logit_p"]).cpu().detach().numpy())
        sp_gamma.append(out["gamma"].cpu().detach().numpy())
        sp_phi.append(out["phi"].cpu().detach().numpy())
        sp_psi0.append(out["psi0"].cpu().detach().numpy())

    p_df = pd.DataFrame(np.concatenate(sp_p), columns=["p_" + y for y in years])
    gamma_df = pd.DataFrame(
        np.concatenate(sp_gamma), columns=["gamma_" + y for y in years[:-1]]
    )
    phi_df = pd.DataFrame(
        np.concatenate(sp_phi), columns=["phi_" + y for y in years[:-1]]
    )
    psi0_df = pd.DataFrame(np.concatenate(sp_psi0), columns=["psi0"])

    res = pd.concat(
        (
            sp_df.filter(
                regex="^[0-9]{4}$|sp.bbs|route_id|Lat|Lon", axis=1
            ).reset_index(),
            p_df,
            gamma_df,
            phi_df,
            psi0_df,
        ),
        axis=1,
    )
    res.to_csv(out_path, index=False, na_rep="NA")

    del sp_net
    del sp_train
    del sp_valid
    del sp_train_loader
    del sp_valid_loader


# iterate over species and fit a model for each
print("Training and saving single-species results")
for sp in tqdm(dataset.cat_ix["english"].keys()):
    fit_single_species_model(sp, model=OneSpeciesNet)

# Training the final model
final_net = MultiNet(
    num_sp=nsp,
    num_gn=ngn,
    num_fm=nfm,
    num_or=nor,
    num_l1=len(dataset.cat_ix["L1_KEY"]),
    nx=nx,
    nt=nt,
    nx_p=nx_p,
)
final_net.to(device)
final_optimizer = optim.Adam(final_net.parameters(), weight_decay=1e-3)
bbs_retrain = dataset.BBSData(dataset.bbs.query("group != 'test'"))
retrain_loader = DataLoader(
    bbs_retrain, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
)
retrain_loss = list()

print("Retraining the final model...")
for epoch in range(n_epoch):
    print("Starting epoch " + str(epoch + 1) + " of " + str(n_epoch) + "...")
    with torch.enable_grad():
        final_net.train()
        retrain_loss.append(
            utils.fit_epoch(
                final_net,
                retrain_loader,
                training=True,
                optimizer=final_optimizer,
                pb=True,
            )
        )

## Save species-specific predictions

weight_dir = os.path.join("out", "weights")
if not os.path.exists(weight_dir):
    os.mkdir(weight_dir)


def melt(a, cols=["row_idx", "t", "h_dim", "value"]):
    """ Convert an nd array to a data frame. """
    res = pd.DataFrame([tuple(list(x) + [val]) for x, val in np.ndenumerate(a)])
    res.columns = cols
    return res


print("Saving output from final model...")
for sp in tqdm(dataset.cat_ix["english"].keys()):
    final_net.eval()
    with torch.no_grad():
        sp_df = dataset.bbs.query("english == @sp")
        years = list(sp_df.filter(regex="^[0-9]{4}", axis=1))
        sp_ds = dataset.BBSData(sp_df)
        sp_loader = DataLoader(
            sp_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        sp_p = []
        sp_gamma = []
        sp_phi = []
        sp_psi0 = []
        h_p = []
        h_phi = []
        h_gamma = []
        h_psi0 = []

        for i_batch, xy in enumerate(sp_loader):
            _, out = utils.bbs_nll(xy, model=final_net)
            sp_p.append(torch.sigmoid(out["logit_p"]).cpu().detach().numpy())
            sp_gamma.append(out["gamma"].cpu().detach().numpy())
            sp_phi.append(out["phi"].cpu().detach().numpy())
            sp_psi0.append(out["psi0"].cpu().detach().numpy())
            h_p.append(out["h_p"].cpu().detach().numpy())
            h_phi.append(out["h_phi"].cpu().detach().numpy())
            h_gamma.append(out["h_gamma"].cpu().detach().numpy())
            h_psi0.append(out["h_psi0"].cpu().detach().numpy())

        p_df = pd.DataFrame(
            np.concatenate(sp_p), columns=["p_" + y for y in years]
        )
        gamma_df = pd.DataFrame(
            np.concatenate(sp_gamma), columns=["gamma_" + y for y in years[:-1]]
        )
        phi_df = pd.DataFrame(
            np.concatenate(sp_phi), columns=["phi_" + y for y in years[:-1]]
        )
        psi0_df = pd.DataFrame(np.concatenate(sp_psi0), columns=["psi0"])

        sp_name_lower = re.sub(" ", "_", sp.lower())
        out_path = f"out/{sp_name_lower}_finalnet.csv"
        res = pd.concat(
            (
                sp_df.filter(
                    regex="^[0-9]{4}$|sp.bbs|route_id|Lat|Lon", axis=1
                ).reset_index(),
                p_df,
                gamma_df,
                phi_df,
                psi0_df,
            ),
            axis=1,
        )
        res.to_csv(out_path, index=False, na_rep="NA")

        # write out weights, which are constant within species
        # (hence we can extract first row ~~ first sample from the minibatch)
        psi0_weight_df = melt(
            out["psi0_weights"][0, :].cpu().detach().numpy(),
            cols=["h_dim", "value"],
        )
        psi0_weight_df["par"] = "psi0"

        p_weight_df = melt(
            out["p_weights"][0, :].cpu().detach().numpy(),
            cols=["h_dim", "value"],
        )
        p_weight_df["par"] = "p"

        phi_weight_df = melt(
            out["phi_weights"][0, :].cpu().detach().numpy(),
            cols=["h_dim", "value"],
        )
        phi_weight_df["par"] = "phi"

        gamma_weight_df = melt(
            out["gamma_weights"][0, :].cpu().detach().numpy(),
            cols=["h_dim", "value"],
        )
        gamma_weight_df["par"] = "gamma"

        weight_df = pd.concat(
            (phi_weight_df, gamma_weight_df, p_weight_df, psi0_weight_df),
            axis=0,
            sort=True,
        )
        weight_df.to_csv(
            os.path.join(weight_dir, f"{sp_name_lower}_weights.csv"),
            index=False,
            na_rep="NA",
        )

# Write out final route vector representations

h_phi_df = melt(
    np.concatenate(h_phi).reshape((-1, final_net.nt - 1, final_net.h_dim))
)
h_phi_df["par"] = "phi"
h_gamma_df = melt(
    np.concatenate(h_gamma).reshape((-1, final_net.nt - 1, final_net.h_dim))
)
h_gamma_df["par"] = "gamma"
h_p_df = melt(np.concatenate(h_p).reshape((-1, final_net.nt, final_net.h_dim)))
h_p_df["par"] = "p"

h_psi0_df = pd.DataFrame(np.concatenate(h_psi0))
h_psi0_df["par"] = "psi0"
h_psi0_df["row_idx"] = list(range(h_psi0_df.shape[0]))

# reshape to permit concatenation
h_psi0_df = pd.melt(h_psi0_df, id_vars=["row_idx", "par"]).rename(
    columns={"variable": "h_dim"}
)

# concatenate all route vectors into one data frame and save
h_df = pd.concat((h_phi_df, h_gamma_df, h_p_df, h_psi0_df), axis=0, sort=True)
h_df.to_csv("out/route_embeddings.csv", index=False)


#############################################################################
#############################################################################
#############################################################################

class MultiNet(nn.Module):
    """ Multispecies neural dynamic occupancy model. """

    embed_dim = 2 ** 6
    h_dim = 2 ** 5
    activation = nn.LeakyReLU()

    def __init__(self, num_sp, num_gn, num_fm, num_or, num_l1, nx, nt, nx_p):
        """Initialize a network object.

        Args
        ----
        num_sp (int): number of species
        num_gn (int): number of genera
        num_fm (int): number of families
        num_or (int): number of orders
        num_l1 (int): number of level 1 ecoregions
        nx (int): number of route-level input features
        nt (int): number of timesteps (years)
        nx_p (int): number of detection/survey-level covariates
        """
        # Define embeddings and fully connected layers
        super(MultiNet, self).__init__()
        self.nt = nt
        self.nx_p = nx_p

        # shared layers
        self.l1_embed = nn.Embedding(num_l1, self.embed_dim)
        self.sp_embed = nn.Embedding(num_sp, self.embed_dim)
        self.gn_embed = nn.Embedding(num_gn, self.embed_dim)
        self.fm_embed = nn.Embedding(num_fm, self.embed_dim)
        self.or_embed = nn.Embedding(num_or, self.embed_dim)

        self.to_h = nn.Sequential(
            nn.Linear(self.embed_dim + nx, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, self.h_dim),
            self.activation,
        )

        # component hidden layers
        self.to_h_psi0 = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim), self.activation
        )

        self.to_h_t = nn.GRU(
            input_size=self.h_dim,
            hidden_size=self.h_dim * 3,
            num_layers=2,
            batch_first=True,
        )

        # weight layers
        self.to_psi0_weights = self.sp_embed_to_w(self.h_dim)
        self.to_p_weights = self.sp_embed_to_w(self.h_dim + nx_p)
        self.to_phi_weights = self.sp_embed_to_w(self.h_dim)
        self.to_gamma_weights = self.sp_embed_to_w(self.h_dim)

    def sp_embed_to_w(self, dim_out):
        """ Mapping from an embedding to a weight vector.

        Args
        ----
        dim_out (int): the dimension of the weight vector
        """
        layers = nn.Sequential(
            nn.Linear(self.embed_dim * 4, self.h_dim),
            self.activation,
            nn.Linear(self.h_dim, dim_out),
        )
        return layers

    def forward(self, sp, gn, fm, order, l1, x, x_p):
        """ Make a forward pass for the model.

        Args
        ----
        sp (tensor): integer indices for species
        gn (tensor): integer indices for genera
        fm (tensor): integer indices for families
        order (tensor): integer indices for orders
        l1 (tensor): integer indices for level 1 ecoregion
        x (tensor): route-level features
        x_p (tensor): survey-level features

        Return
        ------
        A dictionary of parameters
        """

        l1_vec = self.l1_embed(l1)
        sp_vec = torch.cat(
            (
                self.sp_embed(sp),
                self.gn_embed(gn),
                self.fm_embed(fm),
                self.or_embed(order),
            ),
            -1,
        )

        h = self.to_h(torch.cat((l1_vec, x), -1))
        h_psi0 = self.to_h_psi0(h)
        h_t, _ = self.to_h_t(h.view(-1, 1, self.h_dim).repeat(1, self.nt, 1))
        h_t = self.activation(h_t.view(-1, self.nt, self.h_dim, 3))
        h_phi = h_t[:, :-1, :, 0]
        h_gamma = h_t[:, :-1, :, 1]
        h_p = h_t[:, :, :, 2]

        phi_weights = self.to_phi_weights(sp_vec)
        phi = torch.sigmoid(
            torch.bmm(h_phi, phi_weights.view(-1, self.h_dim, 1)).view(
                -1, self.nt - 1
            )
        )

        gamma_weights = self.to_gamma_weights(sp_vec)
        gamma = torch.sigmoid(
            torch.bmm(h_gamma, gamma_weights.view(-1, self.h_dim, 1)).view(
                -1, self.nt - 1
            )
        )

        # for each year, concatenate the route embedding with survey condtion data
        p_weights = self.to_p_weights(sp_vec)
        p_t = list()
        for t in range(self.nt):
            # stack hidden state with survey data
            p_t.append(
                torch.bmm(
                    torch.cat((h_p[:, t, :], x_p[:, t, :]), -1).view(
                        -1, 1, self.h_dim + self.nx_p
                    ),
                    p_weights.view(-1, self.h_dim + self.nx_p, 1),
                ).squeeze()
            )
        logit_p = torch.stack(p_t, dim=-1)

        psi0_weights = self.to_psi0_weights(sp_vec)  # (batch, h_dim)
        psi0 = torch.sigmoid(
            # take the row-wise dot product (route vector * species weights)
            torch.bmm(
                h_psi0.view(-1, 1, self.h_dim),
                psi0_weights.view(-1, self.h_dim, 1),
            )
        ).view(-1, 1)

        out_dict = {
            "phi": phi,
            "gamma": gamma,
            "psi0": psi0,
            "logit_p": logit_p,
            "phi_weights": phi_weights,
            "gamma_weights": gamma_weights,
            "psi0_weights": psi0_weights,
            "p_weights": p_weights,
            "h_p": h_p,
            "h_psi0": h_psi0,
            "h_gamma": h_gamma,
            "h_phi": h_phi,
        }
        return out_dict


# Training the neural network

n_epoch = 5
batch_size = 2 ** 11
print(f"Batch size is {batch_size}")

train_loader = DataLoader(
    bbs_train, batch_size=batch_size, shuffle=True, num_workers=multiprocessing.cpu_count()
)
valid_loader = DataLoader(
    bbs_valid, batch_size=batch_size * 4, num_workers=multiprocessing.cpu_count(), shuffle=True
)

net = MultiNet(
    num_sp=nsp,
    num_gn=ngn,
    num_fm=nfm,
    num_or=nor,
    num_l1=len(dataset.cat_ix["L1_KEY"]),
    nx=nx,
    nt=nt,
    nx_p=nx_p,
)
net.to(device)

optimizer = optim.Adam(net.parameters(), weight_decay=1e-3)

train_loss = list()
valid_loss = list()

print("Training the multispecies model...")
for epoch in range(n_epoch):
    print("Starting epoch " + str(epoch + 1) + " of " + str(n_epoch) + "...")
    with torch.enable_grad():
        net.train()
        train_loss.append(
            utils.fit_epoch(
                net, train_loader, training=True, optimizer=optimizer, pb=True
            )
        )
    with torch.no_grad():
        net.eval()
        valid_loss.append(
            utils.fit_epoch(net, valid_loader, training=False, pb=False)
        )
        print(np.mean(valid_loss[-1]))

print("Validation loss for the multispecies model:")
print([np.mean(l) for l in valid_loss])

# Make and save predictions for each species
print("Writing multispecies output for training & validation data.")
for sp in tqdm(dataset.cat_ix["english"].keys()):
    net.eval()
    with torch.no_grad():
        sp_df = dataset.bbs.query("english == @sp & group != 'test'")
        years = list(sp_df.filter(regex="^[0-9]{4}", axis=1))
        sp_ds = dataset.BBSData(sp_df)
        sp_loader = DataLoader(
            sp_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=multiprocessing.cpu_count(),
            pin_memory=True,
        )

        sp_p = []
        sp_gamma = []
        sp_phi = []
        sp_psi0 = []
        for i_batch, xy in enumerate(sp_loader):
            _, out = utils.bbs_nll(xy, model=net)
            sp_p.append(torch.sigmoid(out["logit_p"]).cpu().detach().numpy())
            sp_gamma.append(out["gamma"].cpu().detach().numpy())
            sp_phi.append(out["phi"].cpu().detach().numpy())
            sp_psi0.append(out["psi0"].cpu().detach().numpy())

        p_df = pd.DataFrame(
            np.concatenate(sp_p), columns=["p_" + y for y in years]
        )
        gamma_df = pd.DataFrame(
            np.concatenate(sp_gamma), columns=["gamma_" + y for y in years[:-1]]
        )
        phi_df = pd.DataFrame(
            np.concatenate(sp_phi), columns=["phi_" + y for y in years[:-1]]
        )
        psi0_df = pd.DataFrame(np.concatenate(sp_psi0), columns=["psi0"])

        sp_name_lower = re.sub(" ", "_", sp.lower())
        out_path = f"out/{sp_name_lower}_nnet.csv"
        res = pd.concat(
            (
                sp_df.filter(
                    regex="^[0-9]{4}$|sp.bbs|route_id|Lat|Lon", axis=1
                ).reset_index(),
                p_df,
                gamma_df,
                phi_df,
                psi0_df,
            ),
            axis=1,
        )
        res.to_csv(out_path, index=False, na_rep="NA")

h_df.to_csv("out/route_embeddings.csv", index=False)













#######################################################################################
#######################################################################################
#######################################################################################
device = "cuda:0"
n = 200
nt = 100
k = 50
x = torch.distributions.uniform.Uniform(low=-1 * torch.ones(n), high=torch.ones(n)).sample()
x, _ = torch.sort(x)
x = x.unsqueeze(1)
z_dist = torch.distributions.normal.Normal(loc=0, scale=1)
torch.manual_seed(0)
p = torch.sigmoid(z_dist.sample() + z_dist.sample() * x + z_dist.sample() * x ** 2 - 4)
phi = torch.sigmoid(z_dist.sample() + z_dist.sample() * x + z_dist.sample() * x ** 2 + 1)
gamma = torch.sigmoid(z_dist.sample() + z_dist.sample() * x + z_dist.sample() * x ** 2 - 2)
psi0 = torch.sigmoid(z_dist.sample() + z_dist.sample() * x + z_dist.sample() * x ** 2 - 3)
plt.scatter(x.numpy(), phi.numpy())
plt.scatter(x.numpy(), gamma.numpy())
plt.scatter(x.numpy(), psi0.numpy())
plt.scatter(x.numpy(), p.numpy())
plt.xlabel('Covariate value')
plt.ylabel('')
plt.ylim(0, 1)
plt.legend(['phi', 'gamma', 'psi0', 'p'])
plt.show()
z0 = torch.distributions.Bernoulli(probs=psi0).sample()
z = torch.zeros((n, nt))
z[:, 0] = z0.squeeze()
for t in range(1, nt):
    psi = z[:, t - 1] * phi.squeeze() + (1. - z[:, t - 1]) * gamma.squeeze()
    z[:, t] = torch.distributions.Bernoulli(probs=psi).sample()
plt.figure(figsize=(7, 10))
plt.imshow(z.numpy(), aspect=.8)
y = torch.distributions.Binomial(total_count=k, probs=z * p).sample()
plt.figure(figsize=(7, 10))
plt.imshow(y.numpy(), aspect=.8)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # affine operations: y = Wx + b
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 4)
        self.nt = nt

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        phi = torch.sigmoid(output[:, [0]].repeat(1, self.nt - 1))
        gamma = torch.sigmoid(output[:, [1]].repeat(1, self.nt - 1))
        psi0 = torch.sigmoid(output[:, [2]])
        logit_p = output[:, [3]].repeat(1, self.nt)

        return  phi, gamma, psi0, logit_p


net = Net().to(device)
running_loss = list()
net

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        num_workers=2)

n_epoch = 100
optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-6)

y_j, psi0_j, gamma_j, phi_j, logit_p_j = y_i, psi0_i, gamma_i, phi_i, logit_p_i
def get_loglik2(y_j, psi0_j, gamma_j, phi_j, logit_p_j):
    """ Calculate log likelihood with scaled forward algorithm

    Some nice detail here: https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
    "Some mathematics for HMM" by Dawei Shen, October 13th, 2008
    """
    batch_size = y_j.shape[0]
    batch_size = 1
    nt = y_j.shape[1]

    pr_y_present = torch.exp(torch.distributions.binomial.Binomial(total_count=k, logits=logit_p_j).log_prob(y_j))
    po = torch.stack((pr_y_present, (y_j == 0).to(device, dtype=torch.float64)), -1)  # nt, 2b
    phi_0_j = torch.cat((psi0_j, 1 - psi0_j), -1)
    Omega = torch.stack((torch.stack((phi_j, 1 - phi_j), -1),
                         torch.stack((gamma_j, 1 - gamma_j), -1)),
                        -2)  # dims: (batch_size, nt-1, 2, 2)
    assert Omega.shape == (batch_size, nt - 1, 2, 2)

    c = list()
    alpha_raw = torch.bmm(,
                          torch.diag_embed(po[:, 0, :], dim1=-2, dim2=-1)
                          )
    c.append((torch.ones(batch_size, 1).to(device) / torch.sum(alpha_raw, dim=-1)).squeeze())
    alpha = c[-1].view(-1, 1, 1) * alpha_raw

    for t in range(nt - 1):
        alpha_raw = torch.bmm(alpha,
                              torch.bmm(
                                  Omega[:, t, :, :],
                                  # batch diagonal
                                  torch.diag_embed(po[:, t + 1, :], dim1=-2, dim2=-1),
                              )
                              )
        c.append((torch.ones(batch_size, 1).to(device) / torch.sum(alpha_raw, dim=-1)).squeeze())
        alpha = c[-1].view(-1, 1, 1) * alpha_raw
    c_stacked = torch.stack(c, -1)
    return -torch.sum(torch.log(c_stacked), dim=-1)


for i in tqdm(range(n_epoch)):
    for i_batch, xy in enumerate(dataloader):
        x_i, y_i = xy
        x_i = x_i.to(device)
        y_i = y_i.to(device)[0:1,]
        optimizer.zero_grad()
        phi_i, gamma_i, psi0_i, logit_p_i, = net(x_i[0:1,])

        # compute the negative log likliehood
        ll = get_loglik2(y_i, psi0_i, gamma_i, phi_i, logit_p_i)
        assert torch.sum(torch.isnan(ll)) < 1
        loss = -torch.mean(ll)
        loss.backward()
        optimizer.step()  # Does the update
        running_loss.append(loss.cpu().data.numpy())
plt.plot([i for i in range(len(running_loss))],
         np.array(running_loss),
         alpha=1)
phi_hat, gamma_hat, psi0_hat, p_hat = net(x.to(device))
plt.scatter(x.numpy(), phi.numpy())
plt.scatter(x.numpy(), phi_hat[:, 0].cpu().detach().numpy(), color='r')
plt.title('Persistence probabilities')
plt.ylim(0, 1)
plt.show()
plt.scatter(x.numpy(), gamma.numpy())
plt.scatter(x.numpy(), gamma_hat[:, 0].cpu().detach().numpy(), color='r')
plt.title('Colonization probabilities')
plt.ylim(0, 1)
plt.show()
plt.scatter(x.numpy(), psi0.numpy())
plt.scatter(x.numpy(), psi0_hat.cpu().detach().numpy(), color='r')
plt.title('Initial occupancy probabilities')
plt.ylim(0, 1)
plt.show()
plt.scatter(x.numpy(), p.numpy())
plt.scatter(x.numpy(), torch.sigmoid(p_hat[:, 0]).cpu().detach().numpy(), color='r')
plt.title('Detection probabilities')
plt.ylim(0, 1)
plt.show()