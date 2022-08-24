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
#
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
nsites = 500
nyears = 20
nsurvy = 2
nobs = nsites * nsurvy * nyears
k = 20


# occupancy transition
phix = torch.distributions.uniform.Uniform(low=-torch.ones(nyears), high=torch.ones(nyears)).sample()
psi = torch.distributions.uniform.Uniform(low=-torch.ones(nyears), high=torch.ones(nyears)).sample()

# observation covariate
obsx = torch.distributions.uniform.Uniform(low=-torch.ones(nobs), high=torch.ones(nobs)).sample()
obsx_ten = obsx.view(nsites,20,2) # site, year, visit

# initial occupancy301

psi0x = torch.distributions.uniform.Uniform(low=-torch.ones(nsites), high=torch.ones(nsites)).sample()
#x, _ = torch.sort(x) # sort variable from lowest to highest

# put into columns
phix, obsx, psi0x = phix.unsqueeze(1),obsx.unsqueeze(1),psi0x.unsqueeze(1)



# create the dataframe
cv1,cv2,cv3,cv4,cv5,cv6 = 0.5,-0.1,0.8,0.5,-0.2,-0.05 # coefficients

# predict persistence
phiy = torch.sigmoid(cv1 + cv2 * phix  + cv3 * phix **2-1)
psi_year = cv1 + -0.9* phix  + 0.09* phix **2
# predict colonization
gamy = torch.sigmoid(cv1 + -0.5*cv2 * phix  + -0.5*cv3 * phix **2 + -0.5*cv3 * phix **3+1)

# predict observation model
py = torch.sigmoid(cv4 + cv5 * obsx + cv6 * obsx**2) # observation
py_arr = py.view(nsites,20,2) # site, year, visit

# predict initial occupancy
psi0y = torch.sigmoid(cv4 + 0.07 * psi0x + 0.07 * psi0x**2) # occupancy

# non dynamic model
psi_sites =  0.7 * psi0x + 0.7 * psi0x**2 # occupancy

#########################################
# predict occupancy for non dynamic model
zs = torch.zeros((nsites,nyears))
psis = torch.zeros((nsites,nyears))
psis[:,0] = torch.sigmoid(psi_year[0].squeeze() + psi_sites[:,0].squeeze())
zs[:, 0] = torch.distributions.Bernoulli(probs=psis[:, 0]).sample().squeeze()
for i in range(1,nyears):
    psis[:,i] = torch.sigmoid(psi_year[i,0].squeeze() + psi_sites[:,0].squeeze())
    zs[:,i] = torch.distributions.Bernoulli(probs = psis[:,i] ).sample().squeeze()


# observed p/a
yobss = torch.zeros((nobs,7))

obs_ys = torch.zeros((nsites,nyears,nsurvy))

# if observation is missing use 0 as the covariate - careful centering!

# observed occupancy
for j in range(0, nsites):
    for i in range(0,nyears):
        for n in range(0,nsurvy):
            z_ix = torch.distributions.Bernoulli(probs=zs[j,i]*py_arr[j,i,n]).sample()
            obs_ys[j,i,n] = z_ix

psi_sites_x = psi0x.repeat_interleave(nyears)
psi_year_x = phix.repeat(nsites,1).squeeze()

site_id = torch.range(0,nsites-1).repeat_interleave(nyears)
year_id = torch.range(0,nyears-1).repeat(nsites)

# covariates for neural occupancy model
site_covariates = torch.hstack((psi_sites_x.unsqueeze(1),
        site_id.unsqueeze(1),
        psi_year_x.unsqueeze(1),
        year_id.unsqueeze(1)))

i=1
j=2

# test it will work in the model
sxi = site_covariates[site_covariates[:,1]==i,:]
sxi = sxi[sxi[:,3]==j,(0,2)]
#########################################
#
# set up occupancies for dynamic model
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
site_id = torch.range(0,nsites-1)
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

#########################################
# visualize relationships between covariates and predict variable

plt.plot(torch.range(1,20), phix.numpy(), color='r')
plt.scatter(phix.numpy(), gamy.numpy(), color='b')
plt.scatter(psi0x.numpy(), psi0y.numpy(), color='b')
plt.scatter(obsx.numpy(), py.numpy(), color='b')
plt.scatter(torch.range(1,20), psi.mean(0).numpy(), color='b')
plt.scatter(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.plot(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.title('Occupancy probabilities')
plt.show()

device = "cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define neural network ###################################################################
#########################################
# non dynamic neural network
class Net1(nn.Module):

    n_p_cov = 1
    n_psi_cov = 2
    h_dim = 64
    nyears = 20
    nvis = 2
    nsites = nsites

    def __init__(self):
        super(Net1, self).__init__()

        # one hidden layer
        self.to_h = nn.Linear(self.n_psi_cov, self.h_dim)

        # component
        self.to_psi = nn.Linear(self.h_dim, 1)
        self.to_p = nn.Linear(self.h_dim+self.n_p_cov, 1)

    def forward(self,sxy ,oxy,p):
        # neural net
        h_out = F.elu(self.to_h(sxy))
        psi_out = torch.sigmoid(self.to_psi(h_out))

        c = -1
        # observation
        for s in oxy:
            oxy_i = s[0,:,:]
            c += 1
            for i in range(0, self.nyears):
                hix = h_out[i,:].clone()
                hx_x =  hix.unsqueeze(0)
                hx_xy = hx_x[0, :]

                for j in range(0, self.nvis):
                    x = oxy_i[i,j]
                    tc = torch.cat((hx_xy, x.unsqueeze(0)))
                    p[c,i,j] = torch.sigmoid(self.to_p(tc))

        return psi_out,p

net_static = Net1()
net_static.to(device)

############################################################################################
# send to co
# res
running_loss = list()

# currently set up for static model
optimizer = optim.Adam(net_static.parameters(), weight_decay=1e-8,lr=0.001)
n_epoch = 100

# needs to put data somewhere
__file__ = 'C:\\Users\\arrgre\\PycharmProjects\\pythonProject\\neural'
# 32 samples loaded into train
# parameters
params = {'batch_size':50,
          'shuffle': True,
          'num_workers': 1}

batch = 50
dataset = TensorDataset(site_id.unsqueeze(1))
dataloader = DataLoader(dataset, **params)
dataloader

#############################################################################
# set up for static model

for i in tqdm(range(n_epoch)): # counter
    for i_batch, xy in enumerate(dataloader):
        xy = xy[0]
        id = xy.long()

        # subset data for example
        oxy = obsx_ten[id, :, :].to(device)
        y_i = obs_ys[id, :, :].to(device)
        y_i.size()
        # occupancy covariates
        b = torch.isin(site_covariates[:, 1],id)
        sxy = site_covariates[b,]
        sxy = sxy.to(device)

        # test effects of missing data
        miss = torch.randint(0,19,(5,)).long()
        y_i[:,0,miss, 0:2] = float('nan')
        miss_data = torch.isnan(y_i)
        no_surveys = torch.Tensor.sum(miss_data, -1) == nsurvy
        oxy[miss_data] = 0
        y_i[miss_data] = 0

        # occupancy covariates
        b = torch.isin(site_covariates[:, 1],id)
        sxy = site_covariates[b,]
        sxy = sxy.to(device)

        optimizer.zero_grad()

        # determine for each example whether we know if z = 1
        definitely_present = (torch.Tensor.sum(y_i,-1) > 0).to(device, dtype=torch.float32)
        maybe_absent = (torch.Tensor.sum(y_i,-1) == 0).to(device, dtype=torch.float32)



        # neural net
        p = torch.zeros(batch,nyears, nsurvy).to(device)

        net_out = net_static(sxy[:,(0,2)],  oxy,  p)

        # neural network outputs
        p = net_out[1]
        psi = net_out[0]


        # list for outputs likelihoods
        psize = p.size()
        plik = torch.zeros((batch,1, nyears, nsurvy))
        plik = plik.to(device)

        for s in range(0,batch):
                for k in range(0, nyears):
                    for j in range(0, nsurvy):
                        plik[s, 0,k, j] = p[s,k, j] ** y_i[s,0 ,k, j] * ((1 - p[s,k, j]) ** (1 - y_i[s,0, k, j]))

        plik[miss_data] = 1
        lp_y_present = torch.log(torch.Tensor.prod(plik, -1))  # log likelihood

        d1 = batch * nyears
        # compute the loss (negative log likelihood)
        lp_present = torch.log(psi.squeeze()) + lp_y_present.view(d1)
        mal = torch.hstack((lp_present.unsqueeze(1), torch.log(1 - psi)))
        lp_maybe_absent = torch.logsumexp(mal,dim=1)
        log_prob = definitely_present.view(d1) * lp_present + maybe_absent.view(d1) * lp_maybe_absent

        log_prob = log_prob[no_surveys.view(d1)==False]


        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().data.numpy())

# increase size of training data set to reduce error rate
    # mean loss

plt.plot(np.arange(len(running_loss)), running_loss, c='k')
plt.xlabel("Number of minibatches")
plt.ylabel("Negative log-likelihood")
plt.show()


p = torch.zeros(nsites, nyears, nsurvy).to(device)
sxy = site_covariates[:, (0, 2)].to(device)
c1 = torch.range(0,99).unsqueeze(1)
oxy = obsx_ten[c1.long(),:,:].to(device)
net_out = net_static(sxy,oxy, p)

psix = net_out[0]
t1 = torch.Tensor.cpu( psix.view(500,20))
t1 = t1.mean(0).detach().numpy()

plt.scatter(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.scatter(torch.range(1,20), t1, color='r')
plt.plot(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.plot(torch.range(1,20), t1, color='r')

# define neural network ###################################################################
#########################################
nsites = 1000
nyears = 20
nsurvy = 2
nobs = nsites * nsurvy * nyears
k = 20


# occupancy transition
psix = torch.distributions.uniform.Uniform(low=-torch.ones(nyears-1), high=torch.ones(nyears-1)).sample()

# observation covariate
obsx = torch.distributions.uniform.Uniform(low=-torch.ones(nobs), high=torch.ones(nobs)).sample()
obsx_ten = obsx.view(nsites,20,2) # site, year, visit

# initial occupancy
psi0x = torch.distributions.uniform.Uniform(low=-torch.ones(nsites), high=torch.ones(nsites)).sample()
#x, _ = torch.sort(x) # sort variable from lowest to highest

# put into columns
psix, obsx, psi0x = psix.unsqueeze(1),obsx.unsqueeze(1),psi0x.unsqueeze(1)



# create the dataframe
cv1 = 0.005 # coefficients


# predict observation model
py = torch.sigmoid(0.12 + 0.2 * obsx + 0.2 * obsx**2) # observation
py_arr = py.view(nsites,20,2) # site, year, visit

# predict initial occupancy
s1 = torch.tensor(1).repeat_interleave(nsites)
m1 = torch.zeros(nsites)
error = torch.distributions.normal.Normal(m1,s1).sample()
psi0y = torch.sigmoid(0.09 + 0.07 * psi0x + 0.07 * psi0x**2 + error.unsqueeze(1)) # occupancy

# predict persistence
s2 = torch.tensor(1).repeat_interleave(nyears-1)
m2 = torch.zeros(nyears-1)
error = torch.distributions.normal.Normal(m2,s2).sample()
psi_year = cv1 + -0.9* psix  + 0.09* psix **2 + error

#########################################
# predict occupancy for non dynamic model
zs = torch.zeros((nsites,nyears))
psis = torch.zeros((nsites,nyears))
psis[:,0] = psi0y.squeeze()
zs[:, 0] = torch.distributions.Bernoulli(probs=psis[:, 0]).sample().squeeze()
for i in range(1,nyears):
    psis[:,i] = torch.sigmoid(psi_year[i-1,0].squeeze() + torch.logit( psis[:,i-1]))
    zs[:,i] = torch.distributions.Bernoulli(probs = psis[:,i] ).sample().squeeze()



# observed p/a
yobss = torch.zeros((nobs,7))

obs_ys = torch.zeros((nsites,nyears,nsurvy))

# if observation is missing use 0 as the covariate - careful centering!

# observed occupancy
for j in range(0, nsites):
    for i in range(0,nyears):
        for n in range(0,nsurvy):
            z_ix = torch.distributions.Bernoulli(probs=zs[j,i]*py_arr[j,i,n]).sample()
            obs_ys[j,i,n] = z_ix

psi_sites_x = psi0x.view(1,nsites,1)
psi_year_x = psix.repeat(nsites,1).view(nsites,19,1)

#########################################
# visualize relationships between covariates and predict variable
plt.scatter(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.plot(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.title('Occupancy probabilities')
plt.show()

device = "cuda:0" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# recurrent neural network - time element included but dynamics are not explicitly modeled
class RNet1(nn.Module):

    n_p_cov = 1
    n_psi_cov = 1
    h_dim = 64
    nyears = 20
    nvis = 2
    nsites = nsites


    def __init__(self):
        super(RNet1, self).__init__()

        # one hidden layer
        self.to_h0 = nn.Linear(self.n_psi_cov, self.h_dim)
        self.to_h1 = nn.Linear(self.h_dim, self.h_dim)
        self.to_hrnn = nn.RNN(input_size = self.n_p_cov,
                              hidden_size = self.h_dim,
                              num_layers = 1,
                              nonlinearity = 'relu',
                              batch_first = True
                              )


        # component
        self.to_psi0 = nn.Linear(self.h_dim, 1)
        self.to_psi = nn.Linear(self.h_dim, 1)
        self.to_p = nn.Linear(self.h_dim+self.n_p_cov, 1)

    def forward(self,sxy0,sxy,oxy,p,xn ):
        # neural net
        hs_out = F.elu(self.to_h0(sxy0))
        h0_out = F.elu(self.to_h1(hs_out))


        # recurrent neural netwok
        h_out,h_n = self.to_hrnn(sxy, h0_out)

        # to occupancy
        psi0_out = torch.sigmoid(self.to_psi0(h0_out))
        psi_out = torch.sigmoid(self.to_psi(h_out))


        # observation
        for i in range(0, xn):

            hi = h_out[i,:,:].clone()
            oi = oxy[i,:,:]

            for j in range(0, self.nvis):

                # first year
                h1 = h0_out[0,i,:].clone()
                o1 = oi[0,j]
                t1 = torch.cat((h1, o1.unsqueeze(0)))
                p[i, 0, j] = torch.sigmoid(self.to_p(t1))

                # the rest of the years
                tc = torch.hstack((hi,oi[1:,j].unsqueeze(1)))
                yn = torch.sigmoid(self.to_p(tc))
                p[i, 1:, j]  = yn.squeeze()

        return psi0_out ,psi_out ,p

# set up test dated to correct network runs properly
id = xy.squeeze().long().tolist()
sxy = psi_year_x[id,:,:].to(device)
sxy.size()
sxy0 = psi_sites_x[:,id,:].to(device)
oxy = obsx_ten[id, :, :].to(device)
xn = oxy.size()[0]
p = torch.zeros(xn, nyears, nsurvy).to(device)

# neural net
net_static = RNet1()
net_static.to(device)
out=net_static(sxy0,sxy,oxy,p,xn)

############################################################################################
# send to co
# res
running_loss = list()

# currently set up for static model
optimizer = optim.Adam(net_static.parameters(), weight_decay=1e-8,lr=0.01)
n_epoch = 100

# needs to put data somewhere
__file__ = 'C:\\Users\\arrgre\\PycharmProjects\\pythonProject\\neural'
# 32 samples loaded into train
# parameters
params = {'batch_size':128,
          'shuffle': True,
          'num_workers': 3}

batch = 128
site_id = torch.range(0,nsites-1).repeat_interleave(nyears)
dataset = TensorDataset(site_id.unsqueeze(1))
dataloader = DataLoader(dataset, **params)
dataloader

#############################################################################

for i in tqdm(range(n_epoch)): # counter
    for i_batch, xy in enumerate(dataloader):
        xy = xy[0]
        id = xy.squeeze().long().tolist()

        # subset data for example
        sxy = psi_year_x[id, :, :].to(device)        # site covariates
        sxy0 = psi_sites_x[:, id, :].to(device) # initial occupancy covariate
        oxy = obsx_ten[id, :, :].to(device) # observation covariance
        y_i = obs_ys[id, :, :].to(device) # observed responds
        y_i.size()

        # test effects of missing data
        miss = torch.randint(0,19,(2,)).long()
        y_i[:,miss, 0:2] = float('nan')
        miss_data = torch.isnan(y_i)
        no_surveys = torch.Tensor.sum(miss_data, -1) == nsurvy
        oxy[miss_data] = 0
        y_i[miss_data] = 0

        optimizer.zero_grad()

        # determine for each example whether we know if z = 1
        definitely_present = (torch.Tensor.sum(y_i,-1) > 0).to(device, dtype=torch.float32)
        maybe_absent = (torch.Tensor.sum(y_i,-1) == 0).to(device, dtype=torch.float32)

        # neural net
        xn = oxy.size()[0]
        p = torch.zeros(xn,nyears, nsurvy).to(device)

        psi0,psi,p = net_static(sxy0,sxy,oxy,p,xn)

        # list for outputs likelihoods
        psize = p.size()
        plik = torch.zeros((xn, nyears, nsurvy))
        plik = plik.to(device)

        for s in range(0,xn):
                for k in range(0, nyears):
                    for j in range(0, nsurvy):
                        plik[s, k, j] = p[s,k, j] ** y_i[s ,k, j] * ((1 - p[s,k, j]) ** (1 - y_i[s, k, j]))

        plik[miss_data] = 1
        lp_y_present = torch.log(torch.Tensor.prod(plik, 2))  # log likelihood

        d1 = xn * nyears

        # output log likelihood
        no_surveys.size()
        lp_y_present.size()

        # present
        log_present = torch.zeros((xn, nyears))
        log_present[:,1 ] = torch.log(psi0[0,:,: ].squeeze()) + lp_y_present[:,1].squeeze()
        log_present[:,1: ] = torch.log(psi.squeeze()) + lp_y_present[:,1:]

        # maybe absent
        log_maybe = torch.zeros((xn, nyears,2))
        log_maybe[:,:,0] =  log_present[:,: ]
        log_maybe[:, 1, 1] = torch.log(1-psi0[0,:,: ].squeeze())
        log_maybe[:, 1:, 1] = torch.log(1-psi.squeeze())
        lp_maybe = torch.logsumexp(log_maybe,dim=2)

        # likelihood
        log_prob = definitely_present.to(device) * log_present.to(device) + maybe_absent.to(device) * lp_maybe.to(device)
        log_prob = log_prob.view(d1)[no_surveys.view(d1)==False]


        loss = -torch.mean(log_prob)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.cpu().data.numpy())

# increase size of training data set to reduce error rate
    # mean loss

plt.plot(np.arange(len(running_loss)), running_loss, c='k')
plt.xlabel("Number of minibatches")
plt.ylabel("Negative log-likelihood")
plt.show()

sxy = psi_year_x.to(device)  # site covariates
sxy0 = psi_sites_x.to(device)  # initial occupancy covariate
oxy = obsx_ten.to(device)  # observation covariance
y_i = obs_ys.to(device)  # observed responds

psi0,psi,p = net_static(sxy0,sxy,oxy,p,xn)


t1 = torch.Tensor.cpu( psi)
t1 = t1.mean(0).detach().numpy()
t1.squeeze()

t2 = torch.Tensor.cpu( psi0)
t2 = t2.mean(1).detach().numpy()
t2.squeeze()

plt.scatter(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.scatter(1, t2.squeeze(), color='r')
plt.scatter(torch.range(2,20), t1.squeeze(), color='r')
plt.plot(torch.range(1,20), psis.mean(0).numpy(), color='b')
plt.plot(torch.range(2,20), t1.squeeze(), color='r')

o1 = torch.Tensor.cpu( psi0).detach()
plt.scatter(psi_sites_x.numpy().squeeze(), o1.numpy().squeeze(), color='r')
plt.scatter(psi_sites_x.numpy().squeeze(), psi0y.squeeze(), color='b')

plt.scatter(psi_year_x[0,0:].squeeze(), t1.squeeze(), color='b')
plt.scatter(psi_year_x[0,0:].squeeze(), psis[:,1:].mean(0).numpy(), color='r')
plt.plot(psi_year_x[0,0:].squeeze(), t1.squeeze(), color='r')
plt.plot(psi_year_x[0,0:].squeeze(), psis[:,1:].mean(0).numpy(), color='r')








