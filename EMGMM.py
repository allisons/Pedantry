#!/usr/bin/env python
# 
# Copyright (c) 2016 Kyle Gorman, Allison Sliter
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# 
# em2gaus.py: Expectation maximization for mixture of two univariate Gaussians
# Kyle Gorman <kgorman@ling.upenn.edu>
# Updated with numpy functions by Allison Sliter allison.sliter@gmail.com
# 
# This code depends on numpy
# 
# See USAGE below for user instructions.    
# 
# The EM procedure was introduced in: 
# 
# A. Dempster, N. Laird, and D. Rubin. 1977. Maximum likelihood from incomplete
# data via the EM algorithm. Journal of the Royal Statistical Society, Series 
# B 39(1): 1-38.
# 
# The update rules for mixtures of two univariate Gaussians are taken from 
# course notes for COGS 502 at the University of Pennsylvania, taught by Mark 
# Liberman and Stephen Isard. This particular code was written for my 2012 
# University of Pennsylvania dissertation, though wasn't ultimately used.
# 
# THIS IS EXPERIMENTAL CODE: PLEASE USE AT YOUR OWN RISK.

from __future__ import division
import numpy as np
from scipy.stats import norm, chi2
from operator import itemgetter

## default parameters
_mix = .5
_mu_min = 0.
_mu_max = 2.
_sigma_min = .1
_sigma_max = 1.
## number of random restarts
_n_iterations = 10
_rand_restarts = 1000

class GaussianMixture(object):
    """
    Class representing mixture of N univariate Gaussians and their EM 
    estimation
    
    data:      iterable of numerical values
    mu_min:    minimum start value for mean
    mu_max:    maximum start value for mean
    sigma_min: minimum start value for sigma
    sigma_max: maximum start value for sigma
    """

    def __init__(self, data, mu_min, mu_max, sigma_min, sigma_max, num_gauss=2):
        self.data = data
        self.num_gauss = num_gauss
        self.gausses = np.array([norm(loc=np.random.uniform(low=mu_min, high=mu_max),  scale=np.random.uniform(low=sigma_min, high=sigma_max)) for _ in xrange(self.num_gauss)])
        self.mix = np.empty(num_gauss, dtype=float)
        self.mix.fill(1/self.num_gauss)
        self.weights = np.zeros((len(data), num_gauss), dtype=float)
    
    
    def set_mix(self, mix):
        assert isinstance(mix, ndarray)
        assert mix.shape == self.gausses
        self.mix = mix
    
    def freshen_loglike(self):
        """
        Updates weights and calculates the log-likelihood of total model
        (For post training evaluation)
        """
        N, n = self.weights.shape
        for i in xrange(N):
            for j in xrange(n):
                self.weights[i, j] = self.gausses[j].pdf(data[i])*self.mix[j]
        self.loglike = np.sum(np.log(np.sum(self.weights, axis=1)))
    
    
    def EMstep(self):
        """
        One full EM cycle, 
        E(stimation step) - update the weight matrix given the data and the current gaussian parameters (and calculate loglike).
        M(aximation step) - given weight matrix and data, update the parameters of the gaussians and the mix.
        """ 
        # compute weights
        N, n = self.weights.shape
        for i in xrange(N):
            for j in xrange(n):
                self.weights[i, j] = self.gausses[j].pdf(data[i])*self.mix[j]
        #calculate loglikelihood of model
        self.loglike = np.sum(np.log(np.sum(self.weights, axis=1)))
        # normalize weights
        for i in xrange(N):
            self.weights[i] /= np.sum(self.weights[i])
        
        # compute next best gaussian parameters    
        for i in xrange(self.num_gauss):
            mean = np.sum((self.weights[:,i]*self.data)/np.sum(self.weights[:,i]))
            std = np.sqrt(np.sum(self.weights[:,i]*((self.data-mean)**2))/np.sum(self.weights[:,i]))
            self.gausses[i] = norm(loc=mean, scale=std)
            self.mix[i] = np.sum(self.weights[:,i])
        #normalize mix
        self.mix /= np.sum(self.mix)
    
    def iterate(self, N=1, verbose=False):
        """
        Perform N iterations, then compute log-likelihood
        """
        for i in xrange(1, N + 1):
            self.EMstep()
            if verbose:
                print '{0:2} {1}'.format(i, self)
    
    def get_params(self):
        params = []
        for i in xrange(self.num_gauss):
            mean = self.gausses[i].mean()
            std = self.gausses[i].std()
            mix = self.mix[i]
            params.append(np.array((mean, std, mix)))
        return sorted(params, key=itemgetter(0))

    def __repr__(self):
        string = "gaussianMixture\nMean\tStd\tWeight\n"
        
        for i in xrange(self.num_gauss):
            m, v = self.gausses[i].stats(moments='mv')
            mix = self.mix[i]
            string += "{0:4.6}\t{1:4.6}\t{2:4.6}\n".format(str(m), str(np.sqrt(v)), str(mix))
        return string
            

    def __str__(self):
        string = "gaussianMixture\nMean\tStd\tWeight\n"
        
        for i in xrange(self.num_gauss):
            m, v = self.gausses[i].stats(moments='mv')
            mix = self.mix[i]
            string += "{0:4.6}\t{1:4.6}\t{2:4.6}\n".format(str(m), str(np.sqrt(v)), str(mix))
        return string

    @classmethod
    def fit(GaussianMixture, data, mu_min, mu_max, sigma_min, sigma_max, num_guassians=2, num_iter=10, num_restarts=100):
        best_gaus = None
        best_loglike = float('-inf')
        for i in xrange(num_restarts):
            mix = GaussianMixture(data, mu_min, mu_max, sigma_min, sigma_max, num_gauss=num_guassians)
            for i in xrange(num_iter):
                try:
                    mix.iterate()
                    if mix.loglike > best_loglike:
                        best_loglike = mix.loglike
                        best_gaus = mix
                except (ZeroDivisionError, ValueError):
                    pass
        
        return best_guas.params()
        
        

if __name__ == '__main__':

    from sys import argv, stderr

    if len(argv) < 3: 
        exit('USAGE: ./em2gaus.py DATA [NUM_GAUSSIANS]')

    ## read in data
    data = [float(d) for d in open(argv[1], 'r')]
    if len(argv)>2:
        num_gaussians = int(argv[2])
    else:
        num_gaussians = 2

    ## compute unimodal model
    uni = norm(loc=np.mean(data),  scale=np.std(data))
    uni_loglike = np.sum(np.log(uni.pdf(d)) for d in data)

    print 'Best singleton: {0:4.6}, {1:4.6}'.format(np.mean(data), np.std(data))
    print 'Null LL: {0:4.6}'.format(uni_loglike)

    ## find best one
    # set defaults
    best_gaus = None
    best_loglike = float('-inf')
    stderr.write('Computing best model with random restarts...\n')
    for i in xrange(_rand_restarts):
        mix = GaussianMixture(data, _mu_min, _mu_max, _sigma_min, _sigma_max, num_gauss=num_gaussians)
        # I catch division errors from bad starts, and just throw them out...
        for i in xrange(_n_iterations):
            try:
                mix.iterate()
                if mix.loglike > best_loglike:
                    best_loglike = mix.loglike
                    best_gaus = mix
            except (ZeroDivisionError, ValueError):
                pass
    print 'Best {0}'.format(best_gaus)
    print 'Alternative LL: {0:4.6}'.format(best_gaus.loglike)
    test_stat = -2 * uni_loglike + 2 * best_gaus.loglike
    print 'Test statistic for LLR (Chi-sq, df=3): {0:4.6}'.format(test_stat)
    print 'P = {0:4.6}'.format(chi2.pdf(test_stat, df=3))