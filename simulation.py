import random
import numpy as np # can be used to create an array
#import tensorflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from scipy import optimize
# import pandas as pd
import csv
import matplotlib as mlp
import matplotlib.pyplot as plt
import sympy as sy # all functions loaded into namespace *

# def is used to define a function
def inv_logit(p):
    return 1 / (1 + sy.exp(-p))

inv_logit_vec = np.vectorize(inv_logit)

def logit(p):
    return float (sy.log(p)- sy.log(1 -p))
#################################################################################
# covariate effect
cov1 = 0.2
cov2 = -0.7

# covariate values
x1 =  np.random.normal(loc=0,scale=2.5,size=300) # data for 300 sites
x2 =  np.random.normal(loc=0,scale=2.5,size=300)
alpha = 0.5 # intercept



# predict occupancy
ypsi = alpha + x1*cov1 + x2*cov2
yprob = inv_logit_vec(ypsi)

yobs = np.random.binomial(1,yprob,len(yprob))

# plot covariates against observations
plt.scatter(x=x1, y=inv_logit_vec(ypsi))
plt.scatter(x=x2, y=inv_logit_vec(ypsi))

# create matrix
ydata = np.column_stack([yobs,x1,x2])




######################################################################
# write two likelihood functions to check they are working correctly
def negative_likelihood(y,yhat):
    log_yhat = np.log(yhat)
    sub_log_yhat = np.log(1 - yhat)
    sub_y = 1 - y
    pos_y = np.multiply(log_yhat,y)
    neg_y = np.multiply(sub_log_yhat,sub_y)
    return print(sum( -(pos_y+neg_y )))

# -(log(yhat_i) * y_i + log(1 – yhat_i) * (1 – y_i))

def loop_likelihood(y,yhat):
    log_likelihood = 0
    for i in range(len(y)):
        log_likelihood += y[i]*np.log(yhat[i]) + (1-y[i])*np.log(1-yhat[i])
    return -log_likelihood

negative_likelihood(yobs,yprob)
loop_likelihood(yobs,yprob)
#######################################################################

#######################################################################
############### simple logistic regression estimation #################
#######################################################################

# covariate effect
covs = [0.2,-0.7]

# covariate values
x1 =  np.random.normal(loc=0,scale=2.5,size=400) # data for 300 sites
x2 =  np.random.normal(loc=0,scale=2.5,size=400)
alpha = 0.5 # intercept



# predict occupancy
ypsi = alpha + x1*cov1 + x2*cov2
yprob = inv_logit_vec(ypsi)

yobs = np.random.binomial(1,yprob,len(yprob))

# plot covariates against observations
plt.scatter(x=x1, y=inv_logit_vec(ypsi))
plt.scatter(x=x2, y=inv_logit_vec(ypsi))

# create matrix
ydata = np.column_stack([yobs,x1,x2])


# setup estimation
# starting values
alpha_est = 0.1
beta_est = [0.1,0.1]

# define the number of epochs with n_epoch and learning_rate
n_epoch = 100
learning_rate = 0.01

# lists for parameter outputs
alpha0, beta1, beta2, log_likelihood_vals = [], [], [], []

# gradient descent - takes longer than optimizes (below)
for epoch in range(n_epoch):
    # initiate counter to execute "if cnt % 9 == 0:" code block
    cnt = 1

    # initialize log-likelihood variable
    log_likelihood = 0
    e = 0.00001

    for x in ydata:
        # predict value
        y_pred = inv_logit(alpha_est + sum( x[1:]*beta_est ))
        y_obs = x[0]

        # calculate gradients
        gradients = (y_pred - y_obs ) * x[1:]
        # gradient ascent algorithm (Figure 8); theta vector
        beta_est = beta_est - (learning_rate * gradients)
        alpha_est = alpha_est - (learning_rate * (y_pred - y_obs))
        # maximizing log-likelihood
        log_likelihood += y_obs * np.log(y_pred) + (1 - y_obs) * np.log(1 - y_pred)

########################################################################
###### write above into a function and chose proper optimizers #########

def log_reg(xi):
    log_likelihood = 0
    for x in ydata:
        # predict value
        y_pred = inv_logit(xi[0] + sum( x[1:]*xi[1:] ))
        y_obs = x[0]
        # maximizing log-likelihood
        log_likelihood += y_obs * np.log(y_pred) + (1 - y_obs) * np.log(1 - y_pred)
    return -log_likelihood

optimize.minimize(log_reg,[0.1,0.1,0.1],method='BFGS')

#######################################################################
############### occupancy logistic regression estimation ##############
#######################################################################

# covariate effect
covs = [0.2,-0.7,0.65]

# covariate values
x1 =  np.random.normal(loc=0,scale=2.5,size=100) # data for 100 sites visited 4 times
x2 =  np.random.normal(loc=0,scale=2.5,size=100)
x3 =  np.random.normal(loc=0,scale=2.5,size=400)
alpha = 0.5 # intercept

sites = np.tile(range(0,100),4) #

# predict true occupancy
ypsi = alpha + x1*covs[0] + x2*covs[1]
yprob = inv_logit_vec(ypsi)
ypop_true = np.random.binomial(1,yprob,len(yprob))

# predict observation model
p = inv_logit_vec(0.1+x3*covs[2])
yobs =  np.random.binomial(1,np.tile (ypop_true,4)*p,400)

# indexes
yn = len(yobs) # number of observations
sn = len(x1) # number of sites

init = np.column_stack([yobs,sites])

# record sites with no detections
occ  = [np.max( init[init[0:,1]==0,0])]
for i in range(1,100):
    occ1 = 1- np.max( init[init[0:,1]==i,0])
    occ.append(occ1)

# create inputs
siv = np.column_stack([np.ones(100),x1,x2])
oiv = np.column_stack([np.ones(400),x3])
ydata = np.column_stack([yobs,sites])

nivn = np.shape(siv)[1] # make sure you get correct coefficients
oivn = np.shape(oiv)[1]
eivn = nivn + oivn
np.savetxt('test_data_y.csv',ydata,delimiter=',')
np.savetxt('test_data_siv.csv',siv,delimiter=',')
np.savetxt('test_data_oiv.csv',oiv,delimiter=',')
xi = [0.1,0.2,0.3,0.4,0.5]
########################################################################
###### write above into a function and chose proper optimizers #########

def log_reg_occ(xi):

    # ecological model #
    psi = inv_logit_vec( siv.dot(xi[0:nivn]))

    # set up likelihoods
    log_likelihood_mod = 0
    site_lik = np.ones(100)

    # observation model #
    y_pred_p = inv_logit_vec(oiv.dot(xi[(nivn):eivn]))
    y_obs = ydata[0:,0]

    # likelihood
    likelihood = (np.power(y_pred_p, y_obs)) * (np.power((1 - y_pred_p), (1 - y_obs)))

    lik = np.column_stack([likelihood, sites]) # make sure each site has id

    for i in range(0, 100):
        site_lik[i] = np.prod(lik[lik[0:,1]==i,0])


    # modified likelihood
    log_likelihood_mod =  np.log(site_lik*psi + occ*(1-psi))

    #log_likelihood_mod += np.log(likelihood )
    return -(np.sum(log_likelihood_mod))


#psi < - psiLinkFunc(X % * % params[1: nOP] + X.offset)
#psi[knownOccLog] < - 1
#pvec < - plogis(V % * % params[(nOP + 1): nP] + V.offset)
#cp < - (pvec ^ yvec) * ((1 - pvec) ^ (1 - yvec))
#cp[navec] < - 1  # so that NA's don't modify likelihood
#cpmat < - matrix(cp, M, J, byrow=TRUE)  #
#loglik < - log(rowProds(cpmat) * psi + nd * (1 - psi))

optimize.minimize(log_reg_occ,[0.1,0.1,0.1,0.1,0.1],method='BFGS' ,options={'disp': True})

def gradient(theta, X, y):
    """
    Compute gradient for logistic regression.

    I/P
    ----------
    X : 2D array where each row represent the training example and each column represent the feature ndarray. Dimension(m x n)
        m= number of training examples
        n= number of features (including X_0 column of ones)
    y : 1D array of labels/target value for each traing example. dimension(1 x m)

    theta : 1D array of fitting parameters or weights. Dimension (1 x n)

    O/P
    -------
    grad: (numpy array)The gradient of the cost with respect to the parameters theta
    """
    m, n = X.shape
    x_dot_theta = X.dot(theta)
    error = inv_logit_vec(x_dot_theta) - y

    grad = 1.0 / m * error.T.dot(X)

    return grad

#######################################################################
############### occupancy neural model ################################
#######################################################################
nsites = 400
nyears = 20
nsurvy = 2
nobs = nsites * nsurvy * nyears
k = 20
phix = torch.distributions.uniform.Uniform(low=-torch.ones(nyears), high=torch.ones(nyears)).sample()

obsx = torch.distributions.uniform.Uniform(low=-torch.ones(nobs), high=torch.ones(nobs)).sample()
obsx_ten = obsx.view(400,20,2) # site, year, visit

psi0x = torch.distributions.uniform.Uniform(low=-torch.ones(nsites), high=torch.ones(nsites)).sample()
#x, _ = torch.sort(x) # sort variable from lowest to highest

# put into columns
phix, obsx, psi0x = phix.unsqueeze(1),obsx.unsqueeze(1),psi0x.unsqueeze(1)



# create the dataframe
cv1,cv2,cv3,cv4,cv5,cv6 = 0.5,-0.1,0.8,0.5,-0.2,-0.05 # coefficients
phiy = torch.sigmoid(cv1 + cv2 * phix  + cv3 * phix **2-1) # observation

gamy = torch.sigmoid(cv1 + -0.5*cv2 * phix  + -0.5*cv3 * phix **2 + -0.5*cv3 * phix **3+1)

py = torch.sigmoid(cv4 + cv5 * obsx + cv6 * obsx**2) # observation
py_arr = py.view(400,20,2) # site, year, visit

psi0y = torch.sigmoid(cv4 + 0.7 * psi0x + 0.7 * psi0x**2) # occupancy

# init occupancy z frame
z = torch.zeros((nsites,nyears))
psi = torch.zeros((nsites,nyears))


z0 = torch.distributions.Bernoulli(probs = psi0y).sample()
z0 = z0.squeeze()
z[:,0]  = z0
psi[:,0] = psi0y.squeeze()
# occupancy loop
for t in range(1, nyears):
    psi[:, t]  = z[:, t - 1] * phiy.squeeze()[t-1] + (1. - z[:, t - 1]) * gamy.squeeze()[t-1]
    z[:, t] = torch.distributions.Bernoulli(probs = psi[:, t] ).sample()

psi.mean(0)
# observed p/a
yobs = torch.zeros((nobs,7))
site_id = torch.range(0,399)
vis_id = torch.range(0,1)
year_id = torch.range(0,19)


obs_y = torch.zeros((nsites,nyears,nsurvy))

# if observation is missing use 0 as the covariate - careful centering

# observed occupancy
for j in range(0, nsites):
    for i in range(0,nyears):
        for n in range(0,nsurvy):
            z_ix = torch.distributions.Bernoulli(probs=z[j,i]*py_arr[j,i,n]).sample()
            obs_y[j,i,n] = z_ix

# visualize relationship

plt.scatter(phix.numpy(), phiy.numpy(), color='r')
plt.scatter(phix.numpy(), gamy.numpy(), color='b')
plt.scatter(psi0x.numpy(), psi0y.numpy(), color='b')
plt.scatter(obsx.numpy(), py.numpy(), color='b')
plt.scatter(torch.range(1,20), psi.mean(0).numpy(), color='b')
plt.plot(torch.range(1,20), psi.mean(0).numpy(), color='b')
plt.title('Occupancy probabilities')
plt.show()

device = "cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# define neural network ###################################################################


class Net(nn.Module):

    n_p_cov = 1
    n_psi_cov = 1
    h_dim = 64
    nyears = 20
    nvis = 2
    def __init__(self):
        super(Net, self).__init__()

        # hidden p, gam, phi
        self.to_h = nn.Linear(1, self.h_dim)
        self.to_hpsi0 = nn.Linear(1, self.h_dim)

        # component
        self.to_phi = nn.Linear(self.h_dim, 1)
        self.to_psi0 = nn.Linear(self.h_dim, 1)
        self.to_gam = nn.Linear(self.h_dim, 1)
        self.to_p = nn.Linear(self.h_dim+self.n_p_cov, 1)

    def forward(self,sxy,sx0 ,oxy,hx,p):

        # set up list for outputs
        phi = list()
        gam = list()


        # neural net
        for i in range(0,self.nyears-1):
            hx[i,:] = F.elu(self.to_h(sxy[i]))

        # initial occupancy
        hx0 = F.elu(self.to_hpsi0(sx0))
        psi0 = torch.sigmoid( self.to_psi0(hx0))

        # phi and gam
        for i in range(0, self.nyears - 1):
            phi.append( torch.sigmoid( self.to_phi(hx[i,:])))
            gam.append( torch.sigmoid( self.to_gam(hx[i,:])))


        # observation
        for i in range(0, self.nyears):

            if i == 0:
                hx_x = hx0.squeeze()
            else:
                hx_x =  hx[i-1,:].unsqueeze(0)
                hx_x = hx_x[0, :]

            for j in range(0, self.nvis):
                x = oxy[i,j]
                tc = torch.cat((hx_x, x.unsqueeze(0)))
                p[i,j] = torch.sigmoid(self.to_p(tc))
        return psi0,phi,gam,p

net1 = Net()
net1.to(device)


############################################################################################
# send to co
# res
running_loss = list()


optimizer = optim.Adam(net1.parameters(), weight_decay=1e-8)
n_epoch = 100

# needs to put data somewhere
__file__ = 'C:\\Users\\arrgre\\PycharmProjects\\pythonProject\\neural'
# 32 samples loaded into train
# parameters
params = {'batch_size': 100,
          'shuffle': True,
          'num_workers': 1}


dataset = TensorDataset(site_id.unsqueeze(1))
dataloader = DataLoader(dataset, **params)
dataloader



# test neural occupancy ##########################################################
for i_batch, xy in enumerate(dataloader):
    xy = xy[0]
    # load a minibatch
    for i in xy.long():
        s = i

        # subset data for example
        oxy = obsx_ten[i,:,:].to(device)
        y_i = obs_y[i, :, :].to(device)
        sx0 = psi0x[s].to(device)
        sxy = phix.to(device)

        # test effects of missing data
        y_i[0,1,0:2] =  -10**6
        miss_data = y_i == -10**6
        no_surveys = torch.Tensor.sum( y_i,-1) < 0
        oxy[miss_data] = 0
        y_i[miss_data] = 0

        # zero out the gradients
        optimizer.zero_grad()

        # neural net
        hx = torch.zeros(nyears - 1, 64).to(device)
        p = torch.zeros(nyears,nsurvy).to(device)
        net_out = net1(sxy, sx0, oxy[0,:,:], hx, p)

        # outputs
        psi0 = net_out[0]
        phi = torch.cat(net_out[1])
        gam = torch.cat(net_out[2])
        p  = net_out[3]

        psize = p.size()
        # list for outputs likelihoods
        plik = torch.zeros((1,psize[0],psize[1]))
        plik = plik.to(device)

        # start with likelihood for observations
        with torch.autograd.detect_anomaly():
            for i in range(0, nyears):
                for j in range(0, nsurvy):
                    plik[0,i,j] = p[i,j]**y_i[0,i,j]  * ((1-p[i,j])**(1-y_i[0,i,j] ))

        # missing values dont impact likelihood
        plik[miss_data] = 1
        lp_y_present = torch.log(torch.Tensor.prod(plik,-1)) # log likelihood

        no = torch.Tensor.sum(y_i,-1)==0

        # look at examples with no species
        po = torch.stack((torch.exp(lp_y_present),no.to(device, dtype=torch.float64),),-1,)
        po

        # iterate over training examples and replace po with 2x2 identity matrix
        # when no surveys are conducted
        po[no_surveys, :] = torch.ones(2,dtype=torch.float64).to(device)
        assert torch.sum(torch.isnan(po)) < 1

        # START HERE #
        phi_0_i = torch.cat((out["psi0"], 1 - out["psi0"]), -1)
        Omega = torch.stack(
            (
                torch.stack((out["phi"], 1 - out["phi"]), -1),
                torch.stack((out["gamma"], 1 - out["gamma"]), -1),
            ),
            -2,  # stacking along rows (so that rows probs to one)
        )
        assert Omega.shape == (batch_size, num_years - 1, 2, 2)


        # initial occupancy
        phi_0_j = torch.cat((psi0, 1 - psi0), -1)

        # colonization and extinction
        Omega = torch.stack((torch.stack((phi, 1 - phi), -1),
                             torch.stack((gam, 1 - gam), -1)),
                            -2)  # dims: (batch_size, nt-1, 2, 2)
        assert Omega.shape == (batch_size, nt - 1, 2, 2)

        c = list()
        alpha_raw = torch.mm(phi_0_j.unsqueeze(1),
                              torch.diag_embed(po[ 0, :], dim1=-2, dim2=-1)
                              )
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

    # determine for each example whether we know if z = 1
    definitely_present = (y_i > 0).to(device, dtype=torch.float32)
    maybe_absent = (y_i == 0).to(device, dtype=torch.float32)
#
    # generate estimates of psi and p from the model
    psi_i, p_i = net1(x_i, x_i)

    # compute the loss (negative log likelihood) currently for binomial
    y_dist_if_present = torch.distributions.binomial.Binomial(total_count=k, probs=p_i)
    lp_present = torch.log(psi_i) + y_dist_if_present.log_prob(y_i) # log likelihood observed and present
    my_abs_l = torch.cat((lp_present, torch.log(1 - psi_i)), dim=1)
    lp_maybe_absent = torch.logsumexp( my_abs_l, dim=1) # log probability if absent
    log_prob = definitely_present * lp_present + maybe_absent * lp_maybe_absent # log probability


    loss = -torch.mean(log_prob) # loss function
    loss.backward()
    optimizer.step()
    running_loss.append(loss.cpu().data.numpy())
######################################################################################################

# train model
for i in tqdm(range(n_epoch)): # counter
    for i_batch, xy in enumerate(dataloader):
        # load a minibatch
        x_i, y_i = xy
        x_i = x_i.to(device)
        y_i = y_i.to(device)

        # zero out the gradients
        optimizer.zero_grad()

        # determine for each example whether we know if z = 1
        definitely_present = (y_i > 0).to(device, dtype=torch.float32)
        maybe_absent = (y_i == 0).to(device, dtype=torch.float32)

        # generate estimates of psi and p from the model
        psi_i, p_i = net1(x_i,x_i)

        # compute the loss (negative log likelihood)
        y_dist_if_present = torch.distributions.binomial.Binomial(total_count=k, probs=p_i)
        lp_present = torch.log(psi_i) + y_dist_if_present.log_prob(y_i) # likelihood of y * probability of present
        lp_maybe_absent = torch.logsumexp(torch.cat((lp_present, torch.log(1 - psi_i)), dim=1),
                                          dim=1)
        log_prob = definitely_present * lp_present + maybe_absent * lp_maybe_absent

        loss = -torch.mean(log_prob) # loss function
        loss.backward() # calculate new weights
        optimizer.step() # updates parameters
        running_loss.append(loss.cpu().data.numpy())
plt.plot(np.arange(len(running_loss)), running_loss, c='k')
plt.xlabel("Number of minibatches")
plt.ylabel("Negative log-likelihood")
plt.show()

psi_hat, p_hat = net1(x.to(device),x.to(device))


plt.scatter(x.numpy(), psi_hat.cpu().detach().numpy(), color='r', alpha=.5)
plt.plot(x.numpy(), psi.numpy(), color='k')
plt.title('Occupancy probabilities')
plt.show()


plt.scatter(x.numpy(), p_hat.cpu().detach().numpy(), alpha=.5, color='r')
plt.plot(x.numpy(), p.numpy(), color='k')
plt.title('Detection probabilities')
plt.ylim(0, 1)


def bbs_nll(xy, model):
    """ Negative log-likelihood for dynamic occupancy model.

    Args
    ----
    xy (tuple): inputs and outputs for the model
    model (torch.nn.Module): a model object to use.

    Returns
    -------
    A tuple of:
    - logliks (torch.tensor): log likelihoods for each example in a minibatch
    - out (dict): output from the model, including parameter values
    """
    sp_i, gn_i, fm_i, or_i, l1_i, x_i, x_p_i, y_i = xy
    sp_i = sp_i.to(device)
    gn_i = gn_i.to(device)
    fm_i = fm_i.to(device)
    or_i = or_i.to(device)
    l1_i = l1_i.to(device)
    x_i = x_i.to(device)
    x_p_i = x_p_i.to(device)
    y_i = y_i.to(device)

    # in cases with no surveys:
    # - set the survey covariates to zero (this does not contribute to loss)
    k_i = torch.ones_like(y_i) * 50  # 50 stops
    no_surveys = y_i != y_i
    x_p_i[no_surveys] = 0
    y_i[no_surveys] = 0

    with torch.autograd.detect_anomaly():
        out = model(sp_i, gn_i, fm_i, or_i, l1_i, x_i, x_p_i)
        batch_size = y_i.shape[0]
        num_years = y_i.shape[1]

        lp_y_present = torch.distributions.binomial.Binomial(
            total_count=k_i, logits=out["logit_p"]
        ).log_prob(y_i)
        po = torch.stack(
            (
                torch.exp(lp_y_present),
                (y_i == 0).to(device, dtype=torch.float64),
            ),
            -1,
        )

        # iterate over training examples and replace po with 2x2 identity matrix
        # when no surveys are conducted
        po[no_surveys, :] = torch.ones(2).to(device)
        assert torch.sum(torch.isnan(po)) < 1

        phi_0_i = torch.cat((out["psi0"], 1 - out["psi0"]), -1)
        Omega = torch.stack(
            (
                torch.stack((out["phi"], 1 - out["phi"]), -1),
                torch.stack((out["gamma"], 1 - out["gamma"]), -1),
            ),
            -2,  # stacking along rows (so that rows probs to one)
        )
        assert Omega.shape == (batch_size, num_years - 1, 2, 2)

        c = list()
        # first year: t = 0
        alpha_raw = torch.bmm(
            phi_0_i.unsqueeze(1),
            # batch diag
            torch.diag_embed(po[:, 0, :], dim1=-2, dim2=-1),
        )
        c.append(
            (
                    torch.ones(batch_size, 1).to(device)
                    / torch.sum(alpha_raw, dim=-1)
            ).squeeze()
        )
        alpha = c[-1].view(-1, 1, 1) * alpha_raw

        # subsequent years: t > 0
        for t in range(num_years - 1):
            tmp = torch.bmm(
                Omega[:, t, :, :],
                # batch diagonal
                torch.diag_embed(po[:, t + 1, :], dim1=-2, dim2=-1),
            )
            alpha_raw = torch.bmm(alpha, tmp)
            c.append(
                (
                        torch.ones(batch_size, 1).to(device)
                        / torch.sum(alpha_raw, dim=-1)
                ).squeeze()
            )
            alpha = c[-1].view(-1, 1, 1) * alpha_raw
        c_stacked = torch.stack(c, -1)
        # log likelihood for each item in the minibatch
        logliks = -torch.sum(torch.log(c_stacked), dim=-1)
    return logliks, out

