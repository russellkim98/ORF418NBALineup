#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:57:02 2018

@author: therealrussellkim
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# In[Functions]:
    
# Knowledge Gradient with Correlated Beliefs (KGCB)

# notation for the following:
# K is the number of alternatives.
# M is the number of time-steps
# K x M stands for a matrix with K rows and M columns

# This function takes in
# mu:     true values for the mean (K x 1)
# mu_0:   prior for the mean (K x 1)
# beta_w: measurement precision (1/lambda(x)) (K x 1)
# cov_m:   initial covariance matrix (K,K)
# m:      how many measurements will be made (scalar)


# And returns
# mu_est:     Final estimates for the means (K x 1)
# oc:         Opportunity cost at each iteration (1 x M)
# choices:    Alternatives picked at each iteration (1 x M)
# mu_est_all:  Estimates at each iteration (K x M)



def kgcb(mu, mu_0, beta_w, cov_m, m):
    kk = len(mu_0) # number of available choices
    mu_est = mu_0.copy()
    oc = []
    choices = []
    mu_est_all = []
    # try the kgcb for M number of times
    for k in range(m):
        # py is the KG for alternatives
        ##THIS IS THE CODE TO COPY FOR THE POLICY##
        py = []
        # loop over all choices
        for iter1 in range(kk):
            a = mu_est.copy()
            b = np.divide(cov_m[iter1], np.sqrt(1/beta_w[iter1]+cov_m[iter1][iter1]))
            kg = EmaxAffine(a,b)
            py.append(kg)
    
        # THIS SHOULD BE x = np.argmax([(82-n)*py[i]+mu_est[i] for i in range(100)]
        
        
        ## THIS STUFF IS FOR UPDATING EQUATIONS

        # max_value is the best estimated value of the KG
        # x is the argument that produces max_value

        # observe the outcome of the decision
        # w_k=mu_k+Z*SigmaW_k where SigmaW is standard deviation of the
        # error for each observation
        w_k = mu[x]+np.random.randn()/np.sqrt(beta_w[x])

        # updating equations for Normal-Normal model with covariance
        addscalar = (w_k - mu_est[x])/(1/beta_w[x] + cov_m[x][x])
        # cov_m_x is the x-th column of the covariance matrix cov_m
        cov_m_x = np.array([row[x] for row in cov_m])
        mu_est = np.add(mu_est, np.multiply(addscalar, cov_m_x))
        cov_m = np.subtract(cov_m, np.divide(np.outer(cov_m_x, cov_m_x), 1/beta_w[x] + cov_m[x][x]))

        # pick the best one to compare OC
        max_choice = np.argmax(mu_est)

        # calculate the opportunity cost
        o_cost = np.max(mu)-mu[max_choice]

        # update the OC vector
        oc.append(o_cost)
        # update the choice vector
        choices.append(x)
        # update the matrix of estimate
        mu_est_all.append(mu_est)
        del py[:]
    return mu_est, oc, choices, mu_est_all



# Get Knowledge Gradient Prior
def Get_kg(mu_0, beta_w, cov_m):
    kk = len(mu_0) # number of available choices
    # py is the KG for alternatives
    py = []
    for iter1 in range(kk):
        a = mu_0
        b = np.divide(cov_m[iter1], np.sqrt(1/beta_w[iter1]+cov_m[iter1][iter1]))
        kg = EmaxAffine(a,b)
        py.append(kg)
    return(py)



# Online Learning ALgorithm
def Online_kgcb(mu, mu_0, beta_w, cov_m, m):
    kk = len(mu_0) # number of available choices
    mu_est = mu_0.copy()
    oc = []
    choices = []
    mu_est_all = []
    # try the kgcb for M number of times
    for k in range(m):
        # py is the KG for alternatives
        py = []
        for iter1 in range(kk):
            a = mu_est.copy()
            b = np.divide(cov_m[iter1], np.sqrt(1/beta_w[iter1]+cov_m[iter1][iter1]))
            #kg = EmaxAffine(a,b)
            online_kg = a[iter1] + (m-k-1)*EmaxAffine(a,b)
            py.append(online_kg)
    
        x = np.argmax(py)

        # max_value is the best estimated value of the KG
        # x is the argument that produces max_value

        # observe the outcome of the decision
        # w_k=mu_k+Z*SigmaW_k where SigmaW is standard deviation of the
        # error for each observation
        w_k = mu[x]+np.random.randn()/np.sqrt(beta_w[x])

        # updating equations for Normal-Normal model with covariance
        addscalar = (w_k - mu_est[x])/(1/beta_w[x] + cov_m[x][x])
        # cov_m_x is the x-th column of the covariance matrix cov_m
        cov_m_x = np.array([row[x] for row in cov_m])
        mu_est = np.add(mu_est, np.multiply(addscalar, cov_m_x))
        cov_m = np.subtract(cov_m, np.divide(np.outer(cov_m_x, cov_m_x), 1/beta_w[x] + cov_m[x][x]))

        # pick the best one to compare OC
        max_choice = np.argmax(mu_est)

        # calculate the opportunity cost
        o_cost = np.max(mu)-mu[max_choice]

        # update the OC vector
        oc.append(o_cost)
        # update the choice vector
        choices.append(x)
        # update the matrix of estimate
        mu_est_all.append(mu_est)
        del py[:]
    return mu_est, oc, choices, mu_est_all
    

# Calculate the KG value defined by
# E[max_x a_x + b_x Z]-max_x a_x, where Z is a standard
# normal random variable and a,b are 1xM input vectors.
def EmaxAffine(a, b):
    a, b = AffineBreakpointsPrep(a, b)

    c, keep = AffineBreakpoints(a, b)
    keep = [int(keep[i]) for i in range(len(keep))]
    a = a[keep]
    b = b[keep]
    c = np.insert(c[np.add(keep, 1)], 0, 0)
    M = len(keep)

    logbdiff = np.log(np.diff(b))

    if M == 1:
        logy = np.log(a)
    elif M >= 2:
        logy = LogSumExp(np.add(logbdiff, LogEI(-np.absolute(c[1:M]))))

    y = np.exp(logy)
    return y


# Prepares vectors for passing to AffineEmaxBreakpoints, changing their
# order and removing elements with duplicate slope.
def AffineBreakpointsPrep(a, b):
    a = np.array(a)
    b = np.array(b)

    # Sort the pairs (a_i,b_i) in ascending order of slope (b_i),
    # breaking ties in slope with the y-intercept (a_i).
    order = np.lexsort((a, b))
    a = a[order]
    b = b[order]

    # Then, from each pair of indices with the b component equal, remove
    # the one with smaller a component.  This code works because the sort
    # above enforced the condition: if b(i) == b(i+1), then a(i) <= a(i+1).
    keep = [i for i in range(len(b) - 1) if b[i] < b[i + 1]]
    keep.append(len(b) - 1)

    # Note that the elements of keep are in ascending order.
    # This makes it so that b(keep) is still sorted in ascending order.
    a = a[keep]
    b = b[keep]
    return a, b


# Inputs are two M-vectors, a and b.
# Requires that the b vector is sorted in increasing order.
# Also requires that the elements of b all be unique.
# This function is used in AffineEmax, and the preparation of generic
# vectors a and b to satisfy the input requirements of this function are
# shown there.

# The output is an (M+1)-vector c and a vector A ("A" is for accept).  Think of
# A as a set which is a subset of {1,...,M}.  This output has the property
# that, for any i in {1,...,M} and any real number z,
#   i \in argmax_j a_j + b_j z
# iff
#   i \in A and z \in [c(j+1),c(i+1)],
#   where j = sup {0,1,...,i-1} \cap A.
def AffineBreakpoints(a, b):
    # Preallocate for speed.  Instead of resizing the array A whenever we add
    # to it or delete from it, we keep it the maximal size, and keep a length
    # indicator Alen telling us how many of its entries are good.  When the
    # function ends, we remove the unused elements from A before passing
    # it.
    M = len(a)
    c = np.array([None] * (M + 1))
    A = np.array([None] * M)

    # Step 0
    c[0] = -float("inf")
    c[1] = float("inf")
    A[0] = 0
    Alen = 0

    for i in range(M - 1):
        c[i + 2] = float("inf")
        while True:
            j = A[Alen]  # jindex = Alen
            c[1 + j] = (a[j] - a[i + 1]) / (b[i + 1] - b[j])
            if Alen > 0 and c[1 + j] < c[1 + A[Alen - 1]]:
                Alen -= 1  # Remove last element j
                # continue in while loop
            else:
                break  # quit while loop
        A[Alen+1] = i + 1
        Alen += 1
    A = A[0:Alen+1]
    return c, A


# Returns the log of E[(s+Z)^+], where s is a constant and Z is a standard
# normal random variable.  For large negative arguments E[(s+Z)^+] function
# is close to 0.  For large positive arguments, the function is close to the
# argument.  For s large enough, s>-10, we use the formula
# E[(s+Z)^+] = s*normcdf(s) + normpdf(s).  For smaller s we use an asymptotic
# approximation based on Mill's ratio.  EI stands for "expected improvement",
# since E[(s+Z)^+] would be the log of the expected improvement by measuring
# an alternative with excess predictive mean s over the best other measured
# alternative, and predictive variance 0.
def LogEI(s):
# Use the asymptotic approximation for these large negative s.  The
# approximation is derived via:
#   s*normcdf(s) + normpdf(s) = normpdf(s)*[1-|s|normcdf(-|s|)/normpdf(s)]
# and noting that normcdf(-|s|)/normpdf(s) is the Mill's ratio at |s|, which is
# asymptotically approximated by |s|/(s^2+1) [Gordon 1941, also documented in
# Frazier,Powell,Dayanik 2009 on page 14].  This gives,
#   s*normcdf(s) + normpdf(s) = normpdf(s)*[1-s^2/(s^2+1)] = normpdf(s)/(s^2+1).
    n = len(s)
    s = np.array(s)
    logy = np.array([None]*n)
    index = [i for i in range(n) if s[i] < -10]
    if len(index) > 0:
        logy[index] = np.subtract(LogNormPDF(s[index]), np.log(np.add(np.power(s[index], 2), 1).astype(float)))
    # Use straightforward routines for s in the more numerically stable region.
    index = [i for i in range(n) if s[i] >= -10]
    if len(index) > 0:
        s_norm_cdf = [norm.cdf(s[i]) for i in index]
        s_norm_pdf = [norm.pdf(s[i]) for i in index]
        logy[index] = np.log(np.add(np.multiply(s[index], s_norm_cdf), s_norm_pdf).astype(float))
    return logy


# logy = LogNormPDF(z)
# Returns the log of the normal pdf evaluated at z.  z can be a vector or a scalar.
def LogNormPDF(z):
    # log of 1/sqrt(2pi)
    cons = -0.5*np.log(2*np.pi)
    logy = cons - np.divide(np.power(z, 2), 2)
    return logy


# function y=LogSumExp(x)
# Computes log(sum(exp(x))) for a vector x, but in a numerically careful way.
def LogSumExp(x):
    xmax = np.max(x)
    diff_max = x-xmax
    y = xmax + np.log(np.sum(np.exp(diff_max.astype(float))))
    return y


# KGCB Run

# THIS PART GENERATES THE TRUTH FOR A GAUSSIAN PROCESS
# SPITS OUT
# x (THE ALTERNATIVES)
# mu_0 (PRIOR BELIEF ABOUT MEAN)
# mu (THE MEAN (TRUTH))
# covM (THE COVARIANCE MATRIX)

# In[Main Function Part a]:
    
# Let's generate a so called Gaussian Process with zero mean on [1,100]
if __name__=='__main__':
    x = list(range(100)) # domain of the problem
    # Prior belief of revenue is 1400 uniformly
    mu_0 = [1400]*100 # our prior beliefs about the mean

    # this is the main parameter (rho is between 0 and 1)
    # you can play around with the values to see what happens as rho changes
    #rho = 0.10
    # We don't really need rho for this new model
    
    # variance of individual components in mu
    var_mu = 400**2

    # generate the covariance matrix
    M = len(x)
    covM = [[var_mu for i in range(M)] for j in range(M)]

    # the matrix will be symmetric
    # instead of looping over all elements, just loop over the first half
    for i in range(M):
        for j in range(i,M):
            if i != j:
                # We update the covariance initial to match the question
                covM[i][j] = var_mu*np.exp(-(np.absolute(i-j)*0.03))
                covM[j][i] = covM[i][j]

    # generate the truth using the formula given in the question
    price_list = list(range(1,101))
    mu = [p*100*np.exp(-0.02*p) for p in price_list]

    # To see what the functions looks like in the domain [1,100]
    plt.figure(1)
    plt.plot(price_list, mu)
    plt.title('True Revenue')
    plt.xlabel('Choices of Price')
    plt.ylabel('Value')
    # In[Part b]:

    # THIS PART RUNS THE KGCB ON THE VALUES GENERATED BY ABOVE

    # Set the measurement variance
    var_w = 400**2
    beta_W = np.multiply(1/var_w, [1]*M)
    # Precision(Beta)=1/Variance
    
    # Compute Knowledge Gradient
    KG_0 = Get_kg(mu_0, beta_W, covM)
    plt.figure(2)
    plt.plot(price_list, KG_0)
    plt.title('Knowledge Gradient of Prior')
    plt.xlabel('Choices')
    plt.ylabel('Knowledge Gradient Value')
    # In[ Part c]:
    # Repeat this 50 times measurement for 20 simulations
    S = 20
    # Also record the price that are tested
    # Run the KGCB on N budget
    N = 50
    
    for i in range(20):
        # The simple function to run KGCB
        mu_est, oc, choices, mu_est_all = kgcb(mu, mu_0, beta_W, covM, N)
        #mu_est_rec.append(mu_est)
        #choices_rec.append(choices)
        plt.figure(3)
        plt.plot(list(range(0,50)), oc)
        plt.title('Opportunity Cost for Each Path')
        plt.xlabel('Time')
        plt.ylabel('Opportunity Cost Value')
        plt.figure(4)
        plt.plot(list(range(0,50)), choices)
        plt.title('Price Tested Each Time')
        plt.xlabel('Time')
        plt.ylabel('Choices of Price')
    # In[Part d]:
    for i in range(20):
        # The simple function to run KGCB
        mu_est, oc, choices, mu_est_all = Online_kgcb(mu, mu_0, beta_W, covM, N)
        #mu_est_rec.append(mu_est)
        #choices_rec.append(choices)
        plt.figure(5)
        plt.plot(list(range(0,50)), oc)
        plt.title('Online KG Opportunity Cost for Each Path')
        plt.xlabel('Time')
        plt.ylabel('Opportunity Cost Value')
        plt.figure(6)
        plt.plot(list(range(0,50)), choices)
        plt.title('Online KG Price Tested Each Time')
        plt.xlabel('Time')
        plt.ylabel('Choices of Price')
        
        
        
    # The following codes are not required for answering Question 1
    # all the plots are saved as .pdf files under the same folder containing the source code
    # Plot the final estimates vs. truth
    fig, ax = plt.subplots()
    ax.plot(x, mu, 'r-', linewidth=2, label='Truth') # 'r-' makes it a red line, LineWidth is the thickness
    ax.plot(x, mu_est, 'b--', lineWidth=2, label='Estimate') # 'b--' makes it a blue dashed line
    legend = ax.legend(loc='best')
    plt.xlabel('Choices')
    plt.ylabel('Expected reward')
    plt.savefig('Final estimates and truth.pdf', format='pdf')
    plt.clf()

    # Plot the estimate at the Kth time step vs. truth
    K = 10
    fig, ax = plt.subplots()
    ax.plot(x, mu, 'r-', linewidth=2, label='Truth')
    ax.plot(x, mu_est_all[K-1], 'b--', lineWidth=2, label='Estimate')
    legend = ax.legend(loc='best')
    plt.xlabel('Choices')
    plt.ylabel('Expected reward')
    plt.savefig('The {:d}-th estimates and truth.pdf'.format(K), format='pdf')
    plt.clf()

    # Plot Opportunity Cost vs Time
    plt.plot(list(range(N)), oc, 'g-', lineWidth=1.5)
    plt.xlabel('Iteration')
    plt.ylabel('Opportunity cost')
    plt.savefig('Opportunity cost.pdf', format='pdf')
    plt.clf()

