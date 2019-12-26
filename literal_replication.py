# -*- coding: utf-8 -*-
"""
   Replication of Zheng Song, Kjetil Storesletten & Fabrizio Zilibotti, 2011.
  "Growing Like China," American Economic Review,  vol. 101(1), pages 196-233.

--- Step by step replication based on their publicly available Matlab files ---

                         Rodolfo G. Campos
                           December, 2019
"""
 
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.io import loadmat

###############################################################################
#                              parameter.m                                    #
###############################################################################

bet = 0.998 # discount factor of workers
bet_E = bet # discount factor of enterpreneurs
r = 0.0175 # world interest rate
sig = 0.5 # the inverse of intertemporal substitution
alp = 0.5 # capital output elasticity
delta = 0.10 # depreciation rate (del in Matlab)
g_n = 0.03 # exogenous population growth
r_soe_ini = 0.093 # initial lending rate for SOEs
ice = 1-r/(r_soe_ini) # iceberg cost

# TFP growth
g_t = 0.038 # exogenous TFP growth
bet = bet*(1+g_t)**(1-sig) # TFP growth adjusted discount factor
bet_E = bet_E*(1+g_t)**(1-sig) # TFP growth adjusted discount factor

# calibration targets
KY_F_E = 2.65 # the ratio of K/Y in the F sector to K/Y in the E sector
rho_r = (r_soe_ini+0.09)/(r/(1-ice)) # the ratio of the rate of return in the E sector to that in the F sector
psi = 1-(rho_r*r/(1-ice)+delta)/(r/(1-ice)+delta)/KY_F_E # share of managerial compensation
ksi = (KY_F_E)**(alp/(1-alp))/(1-psi) # productivity ratio of E over F

# bank loan in the E sector
loan_asset = 1 # loan asset ratio
eta = loan_asset*(1+r/(1-ice))/(1+rho_r*r/(1-ice)+(rho_r*r/(1-ice)-r/(1-ice))*loan_asset) # measure of financial frictions


#**************************************************************************
# note that eta in the code = eta in the paper / (1 - eta in the paper)
# so eta in the paper = eta in the code / (1 + eta in the code) = 0.46
#**************************************************************************

# initial asset
initial_ratio = 0.80
initial_ratio_E = 0.33

# demographic structure
age_max = 50 # maximum age
age_T = 26 # the age when entrepreneurs become firm owners
age_T_w = 31 # the age when workers retire
time_max = 400 # the end of the economy
n_pre = 100 # the initial size of workers
e_pre = 5 # the initial size of entrepreneurs
# computing demographic structure

n_weight = np.zeros(age_max)
e_weight = np.zeros(age_max)
if g_n > 0 or g_n < 0:
    sum_demo = (1-(1+g_n)**age_max)/(1-(1+g_n)) # sum of total population
    for i in range(age_max):
        n_weight[i]=(1+g_n)**(age_max-i-1)/sum_demo*n_pre # weight
        e_weight[i]=(1+g_n)**(age_max-i-1)/sum_demo*e_pre # weight
else:
    for i in range(age_max):
        n_weight[i] = 1/age_max*n_pre
        e_weight[i] = 1/age_max*e_pre
        
# the initial size of workers before retirement
nw_pre = np.sum(n_weight[0:age_T_w-1])
# the initial size of entrepreneurs after being firm owner
ee_pre = np.sum(e_weight[age_T-1:age_max])

# capital deepening (reform in the financial sector)
time_ab = 9 # beginning of reform
time_cd = 27 # end of reform

ice_t = np.zeros(time_max+age_max-1)
ice_t[:time_ab] = ice # iceberg costs before reform
ice_t[time_cd-1:] = 0.0 # iceberg costs after reform
speed = 2.38 # speed of reform
for t in range(time_ab-1,time_cd-1):
    ice_t[t] = ice+(t-(time_ab-1))**speed*(ice_t[time_cd-1]-ice)/(time_cd-time_ab)**speed

# check parameter values
print('Assumption 1: > 0')
print(ksi-(1/(1-psi))**(1/(1-alp)))

###############################################################################
#                            pre_transition.m                                 #
###############################################################################

k_pre = (alp/(r/(1-ice)+delta))**(1/(1-alp))*(nw_pre) # total capital during pre-transition period (all in the F sector)
w_pre = (1-alp)*(alp/(r/(1-ice)+delta))**(alp/(1-alp)) # wage rate during pre-transition period


def saving_pre_transition(age, wage, wealth):
    """
    Saving rate, consumption and wealth for workers in the pre-transition
    """
    # generating interest rate adjusted life-cycle earnings and others
    w = np.zeros(age_max)
    ratio = np.zeros(age_max)    
    for i in range(age,age_max+1):
        if i < age_T_w:
            w[i-1] = wage*((1+g_t)/(1+r))**(i-age) # earnings
        else:
            w[i-1] = 0.0
    # Life-time wealth (evaluated at t+1)
    A = np.sum(w)+wealth*(1+r)
    # Current optimal consumption and savings
    for i in range(age,age_max+1):
        # the interest rate adjusted ratio of optimal consumption to consumption of the current age
        if i == age:
            ratio[i-1] = 1.0
        else:
            ratio[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*(1+g_t)/(1+r)*ratio[i-2]
    # optimal consumption and savings
    consumption = A/(np.sum(ratio))
    savings = wealth*r+wage-consumption
    sr = savings/(wealth*r+wage) # saving rate
    # computing next-period wealth
    wealth_prime = (wealth*(1+r)+wage-consumption)/(1+g_t)
    return savings, wealth_prime, sr, consumption


def saving_pre_transition_E(age, wage, wealth):
    """
    Saving rate, consumption and wealth for entrepreneurs in the pre-transition
    Note: same functions as for workers, except for the beta
    """
    # generating interest rate adjusted life-cycle earnings and others
    w = np.zeros(age_max)
    ratio = np.zeros(age_max)    
    for i in range(age,age_max+1):
        if i < age_T_w:
            w[i-1] = wage*((1+g_t)/(1+r))**(i-age) # earnings
        else:
            w[i-1] = 0.0
    # Life-time wealth (evaluated at t+1)
    A = np.sum(w)+wealth*(1+r)
    # Current optimal consumption and savings
    for i in range(age,age_max+1):
        # the interest rate adjusted ratio of optimal consumption to consumption of the current age
        if i == age:
            ratio[i-1] = 1.0
        else:
            ratio[i-1] = (bet_E*(1+r)/(1+g_t))**(1/sig)*(1+g_t)/(1+r)*ratio[i-2]
    # optimal consumption and savings
    consumption = A/(np.sum(ratio))
    savings = wealth*r+wage-consumption
    sr = savings/(wealth*r+wage) # saving rate
    # computing next-period wealth
    wealth_prime = (wealth*(1+r)+wage-consumption)/(1+g_t)
    return savings, wealth_prime, sr, consumption

###############
# for workers #
###############

# Life-cycle patterns
wealth_pre = np.zeros(age_max)
consumption_pre = np.zeros(age_max)
sr_pre = np.zeros(age_max)
for i in range(age_max):
    age = i+1
    if age < age_T_w:
        wage = w_pre
    else:
        wage = 0.0 # wage after retirement
    if age == 1: # born without assets
        wealth_pre[i] = 0.0 # wealth
    wealth = wealth_pre[i]
    (_, wealth_prime, sr_pre[i], consumption_pre[i]) = saving_pre_transition(age, wage, wealth)
    if age < age_max:
        wealth_pre[i+1] = wealth_prime

# initial condition
wealth_pre = initial_ratio*wealth_pre

# Compare with Matlab data
data_pre = loadmat(os.path.join('original_files', 'data_pre.mat'))
wealth_pre_2 = data_pre['wealth_pre'].flatten()
check_1 = pd.DataFrame({'Python': wealth_pre, 'Matlab': wealth_pre_2})
check_1.plot(title='wealth_pre', style=['y-','k--'])

#####################
# for entrepreneurs #
#####################

# life-cycle patterns
wealth_pre_E = np.zeros(age_max)
consumption_pre_E = np.zeros(age_max)
sr_pre_E = np.zeros(age_max)
for i in range(age_max):
    age = i+1
    if age < age_T_w:
        wage = w_pre
    else:
        wage = 0.0 # wage after retirement
    if age == 1: # born without assets
        wealth_pre_E[i] = 0.0 # wealth
    wealth = wealth_pre_E[i]
    (_, wealth_prime, sr_pre_E[i], consumption_pre_E[i]) = saving_pre_transition_E(age, wage, wealth)
    if age < age_max:
        wealth_pre_E[i+1] = wealth_prime

# initial condition
wealth_pre_E = initial_ratio_E*wealth_pre_E

# Compare with Matlab data
data_pre_E = loadmat(os.path.join('original_files','data_pre_E.mat'))
wealth_pre_E_2 = data_pre_E['wealth_pre_E'].flatten()
check_1 = pd.DataFrame({'Python': wealth_pre_E, 'Matlab': wealth_pre_E_2})
check_1.plot(title='wealth_pre_E', style=['y-','k--'])

###############################################################################
#                             transition.m                                    #
###############################################################################

# iteration choices
relax = 0.75
iter_max = 1000
tol = 1e-4
dev_max = 1
iteration = 0

# initial guess
# true results
initial_guess = loadmat(os.path.join('original_files','data_result.mat'))

w_t = initial_guess['w_t'].flatten()
m_t = initial_guess['m_t'].flatten()
rho_t = initial_guess['rho_t'].flatten()

######################
#  start to iterate  #
######################

def saving_E_existing(age, current_wealth):
    """
    Savings, wealth and consumption choices of existing entrepreneurs
    """
    # adjusting rate of return due to the endogenous borrowing constraint
    w = np.zeros(age_max)
    ratio = np.zeros(age_max)
    wealth = np.zeros(age_max)
    consumption = np.zeros(age_max)
    wealth[age-1] = current_wealth
    rho_t_ad = np.maximum(rho_t,(rho_t*(1+r/(1-ice_t))+eta*(rho_t-r/(1-ice_t)))/(1+r/(1-ice_t)-eta*(rho_t-r/(1-ice_t))))
    for i in range(age, age_max+1):
        if i < age_T:
            w[i-1] = m_t[i-age]*((1+g_t)/(1+r))**(i-age) # earnings
        else:
            w[i-1] = 0.0
    if age < age_T:
        A = np.sum(w)+(1+r)*wealth[age-1] # Check how wealth is defined
    else:
        A = np.sum(w)+(1+rho_t_ad[0])*wealth[age-1]
    
    # computing current optimal consumption and savings
    for i in range (age,age_max+1):
    # the interest rate adjusted ratio of optimal consumption to consumption of the current age
        if i == age:
            ratio[i-1] = 1.0
        elif i < age_T: # being manager
            ratio[i-1] = (bet_E*(1+r)/(1+g_t))**(1/sig)*(1+g_t)/(1+r)*ratio[i-2]
        else: # become firm owner
            ratio[i-1] = (bet_E*(1+rho_t_ad[i-age])/(1+g_t))**(1/sig)*(1+g_t)/(1+rho_t_ad[i-age])*ratio[i-2]
            
    # optimal consumption and savings
    for i in range(age,age_max+1):
        if i == age:
            consumption[i-1] = A/(np.sum(ratio))
            if i < age_T:
                wealth[i] = (wealth[i-1]*(1+r)+m_t[i-age]-consumption[i-1])/(1+g_t)
            elif i < age_max:
                wealth[i] = (wealth[i-1]*(1+rho_t_ad[i-age])-consumption[i-1])/(1+g_t)
        elif i < age_T: # being manager
            consumption[i-1] = (bet_E*(1+r)/(1+g_t))**(1/sig)*consumption[i-2]
            wealth[i] = (wealth[i-1]*(1+r)+m_t[i-age]-consumption[i-1])/(1+g_t)
        else: # become firm owner
            consumption[i-1] = (bet_E*(1+rho_t_ad[i-age])/(1+g_t))**(1/sig)*consumption[i-2]
            if i < age_max:
                wealth[i] = (wealth[i-1]*(1+rho_t_ad[i-age])-consumption[i-1])/(1+g_t)

    return wealth, consumption


def saving_E_newly_born(year_birth):
    """
    Savings, wealth and consumption choices of newly-born entrepreneurs
    """
    # adjusting rate of return due to the endogenous borrowing constraint
    w = np.zeros(age_max)
    ratio = np.zeros(age_max)
    wealth = np.zeros(age_max)
    consumption = np.zeros(age_max)
    tt = year_birth
    wealth[0] = 0.0
    rho_t_ad = np.maximum(rho_t,(rho_t*(1+r/(1-ice_t))+eta*(rho_t-r/(1-ice_t)))/(1+r/(1-ice_t)-eta*(rho_t-r/(1-ice_t))))
    for i in range(1, age_max+1):
        if i < age_T:
            w[i-1] = m_t[tt+i-2]*((1+g_t)/(1+r))**(i-1) # earnings
        else:
            w[i-1] = 0.0
    A = np.sum(w)
   
    # computing current optimal consumption and savings
    for i in range (1,age_max+1):
    # the interest rate adjusted ratio of optimal consumption to consumption of the current age
        if i == 1:
            ratio[i-1] = 1.0
        elif i < age_T: # being manager
            ratio[i-1] = (bet_E*(1+r)/(1+g_t))**(1/sig)*(1+g_t)/(1+r)*ratio[i-2]
        else: # become firm owner
            ratio[i-1] = (bet_E*(1+rho_t_ad[tt+i-2])/(1+g_t))**(1/sig)*(1+g_t)/(1+rho_t_ad[tt+i-2])*ratio[i-2]
            
    # optimal consumption and savings
    for i in range(1,age_max+1):
        if i == 1:
            consumption[i-1] = A/(np.sum(ratio))
            wealth[i] = (m_t[tt-1]-consumption[i-1])/(1+g_t)
        elif i < age_T: # being manager
            consumption[i-1] = (bet_E*(1+r)/(1+g_t))**(1/sig)*consumption[i-2]
            wealth[i] = (wealth[i-1]*(1+r)+m_t[tt+i-2]-consumption[i-1])/(1+g_t)
        else: # become firm owner
            consumption[i-1] = (bet_E*(1+rho_t_ad[tt+i-2])/(1+g_t))**(1/sig)*consumption[i-2]
            if i < age_max:
                wealth[i] = (wealth[i-1]*(1+rho_t_ad[tt+i-2])-consumption[i-1])/(1+g_t)
    return wealth, consumption




while dev_max > tol and iteration < iter_max:
    # an indicator for the end of transition
    I_end = 0
    
    # Initialize all vectors and matrices
    wealth_E = np.zeros((time_max+age_max-1,age_max))
    consumption_E = np.zeros((time_max+age_max-1,age_max))
    ae = np.zeros((time_max,age))
    AE = np.zeros((time_max,age))
    loan_ratio = np.zeros(time_max)
    loan = np.zeros((time_max,age))
    ke = np.zeros((time_max,age))
    ne = np.zeros((time_max,age))
    KE = np.zeros((time_max,age))
    NE = np.zeros((time_max,age))
    LE = np.zeros((time_max,age))
    AE_t = np.zeros(time_max)
    NE_t = np.zeros(time_max)
    KE_t = np.zeros(time_max)
    LE_t = np.zeros(time_max)
    N_t = np.zeros(time_max)
    YE_t = np.zeros(time_max)
    M_t = np.zeros(time_max)
    w_t_new = np.zeros(time_max+age_max-1)
    rho_t_new = np.zeros(time_max+age_max-1)
    m_t_new = np.zeros(time_max+age_max-1)

    # existing entrepreneurs
    for age in range(2,age_max+1):  # includes entrepreneurs aged age_max but not aged one (they would be new)
        ii = age-1
        # computing existing entrepreneurs' wealth given the guess of  m_t and rho_t
        #y=feval('fun_saving_E_existing',[ii,wealth_pre_E(ii)]);
        wealth, consumption = saving_E_existing(age,wealth_pre_E[ii])
        # wealth and cons time series for the existing enterpreneurs
        for tt in range(age_max-ii):
            wealth_E[tt,ii+tt] = wealth[ii+tt]
            consumption_E[tt,ii+tt] = consumption[ii+tt]
            
    # newly-born entrepreneurs
    for tt in range(time_max):       
        age = tt+1
        wealth, consumption = saving_E_newly_born(age)
        # wealth and cons time series for the existing enterpreneurs
        for ii in range(age_max):
            wealth_E[tt+ii,ii] = wealth[ii]
            consumption_E[tt+ii,ii] = consumption[ii]
 
    # Update new factor price time series
    for t in range(time_max):
        
        # Fixed size of managers
        E_t = e_pre-ee_pre
        
        # Assets in the E sector
        for i in range(age_max):
            
            ae[t,i] = wealth_E[t,i]            # entrepreneurial capital owned by an entprepreneur at time t with age i
            AE[t,i] = e_weight[i]*ae[t,i]    # total capital owned by all entrepreneurs at time t with age i
        
        #capital and labor in the E sector
        for i in range(age_T-1,age_max):
            if rho_t[t] >= r/(1-ice_t[t]):   # borrowing is profitable
                loan_ratio[t] = eta*(1+rho_t[t])/(1+r/(1-ice_t[t])-eta*(rho_t[t]-r/(1-ice_t[t]))) # loan asset ratio
                loan[t,i] = wealth_E[t,i]*loan_ratio[t]
                ke[t,i] = wealth_E[t,i]+loan[t,i] # entrepreneurial capital owned by an entrepreneur at time t with age i
            else: # borrowing is not profitable
                loan[t,i] = 0.0
                ke[t,i] = wealth_E(t,i) # entrepreneurial capital owned by an entrepreneur at time t with age i
                
            ne[t,i] = ke[t,i]*((1-alp)*(1-psi)*ksi**(1-alp)/w_t[t])**(1/alp) #  labor employed by an entrepreneur at time with age i
            KE[t,i] = e_weight[i]*ke[t,i] # total capital owned by all entrepreneurs at time with age i
            NE[t,i] = e_weight[i]*ne[t,i] # total labor employed by all entrepreneurs at time with age i
            LE[t,i] = e_weight[i]*loan[t,i] # total loan

        # resource allocation
        AE_t[t] = AE[t,:].sum() # aggregate capital in the E sector
        NE_t[t] = NE[t,:].sum() # aggregate employment in the E sector
        KE_t[t] = KE[t,:].sum() # when rho > r
        LE_t[t] = LE[t,:].sum() # total loan
        N_t[t] = nw_pre # the size of workers (no migration)   
        
        # factor prices
        w_t_new[t] = (1-psi)*(1-alp)*(KE_t[t]/NE_t[t])**alp*ksi**(1-alp) # wage rate
        
        # locate the end of the transition
        if NE_t[t] >= N_t[t] and I_end == 0:
            I_end = 1
            I_t = t
        elif I_end == 1:
            I_end = 1

        if I_end == 0:
            w_t_new[t] = (1-alp)*(alp/(r/(1-ice_t[t])+delta))**(alp/(1-alp)) # wage rate
        else:
            NE_t[t] = N_t[t]
            w_t_new[t] = (1-psi)*(1-alp)*(KE_t[t]/N_t[t])**alp*ksi**(1-alp) # wage rate

        rho_t_new[t] = np.max([r,(1-psi)**(1/alp)*ksi**((1-alp)/alp)*((1-alp)/w_t_new[t])**((1-alp)/alp)*alp-delta]) # the internal rate of return for entrepreneurs
        YE_t[t] = KE_t[t]**alp*(ksi*NE_t[t])**(1-alp) # aggregate output in the E sector
        M_t[t] = psi*YE_t[t] # total managerial compensations
        m_t_new[t] = M_t[t]/E_t # compensations for young entrepreneurs

    # steady state assumption
    w_t_new[time_max:] = w_t_new[time_max-1]
    rho_t_new[time_max:] = rho_t_new[time_max-1]
    m_t_new[time_max:] = m_t_new[time_max-1]
    
    # deviation
    dev_w = np.abs(w_t_new-w_t)
    dev_rho = np.abs(rho_t_new-rho_t)
    dev_m = np.abs(m_t_new-m_t)
    dev_w_max = dev_w.max()
    dev_rho_max = dev_rho.max()
    dev_m_max = dev_m.max()
    dev_max = np.array([dev_w_max,dev_rho_max,dev_m_max]).max()

    # renew
    w_t = w_t*relax+w_t_new*(1-relax)
    rho_t = rho_t*relax+rho_t_new*(1-relax)
    m_t = m_t*relax+m_t_new*(1-relax)
    
    if int(5*np.floor(iteration/5)) == iteration:
        print("Iteration: {0}, Max deviation: {1}".format(iteration, dev_max))

    iteration += 1

###############################################################################
#                              six_panel.m                                    #
###############################################################################
def saving_F_existing(age, current_wealth):
    """
    Savings, wealth and consumption choices of F agents
    """
    w = np.zeros(age_max)
    ratio = np.zeros(age_max)
    wealth = np.zeros(age_max)
    consumption = np.zeros(age_max)
    wealth[age-1] = current_wealth
    for i in range(age, age_max+1):
        if i < age_T_w:
            w[i-1] = w_t[i-age]*((1+g_t)/(1+r))**(i-age) # earnings
        else:
            w[i-1] = 0.0

    A = np.sum(w)+(1+r)*wealth[age-1] # Check how wealth is defined
    
    # computing current optimal consumption and savings
    for i in range (age,age_max+1):
    # the interest rate adjusted ratio of optimal consumption to consumption of the current age
        if i == age:
            ratio[i-1] = 1.0
        else:
            ratio[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*(1+g_t)/(1+r)*ratio[i-2]
            
    # optimal consumption and savings
    for i in range(age,age_max+1):
        if i == age:
            consumption[i-1] = A/(np.sum(ratio))
            if i < age_T_w:
                wealth[i] = (wealth[i-1]*(1+r)+w_t[i-age]-consumption[i-1])/(1+g_t)
            elif i < age_max:
                wealth[i] = (wealth[i-1]*(1+r)-consumption[i-1])/(1+g_t)
        elif i < age_T_w: # being worker
            consumption[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*consumption[i-2]
            wealth[i] = (wealth[i-1]*(1+r)+w_t[i-age]-consumption[i-1])/(1+g_t)
        else: # become retirees
            consumption[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*consumption[i-2]
            if i < age_max:
                wealth[i] = (wealth[i-1]*(1+r)-consumption[i-1])/(1+g_t)

    return wealth, consumption

def saving_F_newly_born(year_birth):
    """
    Savings, wealth and consumption choices of newly-born F agents
    """
    w = np.zeros(age_max)
    ratio = np.zeros(age_max)
    wealth = np.zeros(age_max)
    consumption = np.zeros(age_max)
    tt = year_birth
    wealth[0] = 0.0
    for i in range(1, age_max+1):
        if i < age_T_w:
            w[i-1] = w_t[tt+i-2]*((1+g_t)/(1+r))**(i-1) # earnings
        else:
            w[i-1] = 0.0
    A = np.sum(w)
   
    # computing current optimal consumption and savings
    for i in range (1,age_max+1):
    # the interest rate adjusted ratio of optimal consumption to consumption of the current age
        if i == 1:
            ratio[i-1] = 1.0
        else:
            ratio[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*(1+g_t)/(1+r)*ratio[i-2]
            
    # optimal consumption and savings
    for i in range(1,age_max+1):
        if i == 1:
            consumption[i-1] = A/(np.sum(ratio))
            wealth[i] = (w_t[tt-1]-consumption[i-1])/(1+g_t)
        elif i < age_T_w: # being workers
            consumption[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*consumption[i-2]
            wealth[i] = (wealth[i-1]*(1+r)+w_t[tt+i-2]-consumption[i-1])/(1+g_t)
        else: # become retirees
            consumption[i-1] = (bet*(1+r)/(1+g_t))**(1/sig)*consumption[i-2]
            if i < age_max:
                wealth[i] = (wealth[i-1]*(1+r)-consumption[i-1])/(1+g_t)
    return wealth, consumption


# Initialize all vectors and matrices
wealth_F = np.zeros((time_max+age_max-1,age_max))
consumption_F = np.zeros((time_max+age_max-1,age_max))
N_t = np.zeros(time_max)
AF = np.zeros((time_max,age))
CF = np.zeros((time_max,age))
CE = np.zeros((time_max,age))
AF_t = np.zeros(time_max)
CF_t = np.zeros(time_max)
CE_t = np.zeros(time_max)
KF_t = np.zeros(time_max)
YF_t = np.zeros(time_max)
NF_t = np.zeros(time_max)
NE_N_t = np.zeros(time_max)
IF_t = np.zeros(time_max)
IE_t = np.zeros(time_max)
IF_Y_t = np.zeros(time_max)
IE_Y_t = np.zeros(time_max)
SF_t = np.zeros(time_max)
SF_YF_t = np.zeros(time_max)
SE_t = np.zeros(time_max)
SE_YE_t = np.zeros(time_max)
Y_N_t = np.zeros(time_max)
I_Y_t = np.zeros(time_max)
S_Y_t = np.zeros(time_max)
K_Y_t = np.zeros(time_max)
FA_Y_t = np.zeros(time_max)
BoP_Y_t = np.zeros(time_max)
TFP_t = np.zeros(time_max)
YG_t = np.zeros(time_max)

# workers' savings and assets
for age in range(2,age_max+1):
    ii = age-1
    # computing existing workers' wealth given the guess of m_t and rho_t
    wealth, consumption = saving_F_existing(age,wealth_pre[ii])
    # wealth and cons time series for the existing workers
    for tt in range(age_max-ii):
        wealth_F[tt,ii+tt] = wealth[ii+tt]
        consumption_F[tt,ii+tt] = consumption[ii+tt]

# newly-born workers
for tt in range(time_max):       
    age = tt+1
    wealth, consumption = saving_F_newly_born(age)
    # wealth and cons time series for the existing workers
    for ii in range(age_max):
        wealth_F[tt+ii,ii] = wealth[ii]
        consumption_F[tt+ii,ii] = consumption[ii]


# demographic structure and others
for t in range(time_max):
    
    # no migration
    N_t[t] = nw_pre
    
    # total assets of workers and total consumptions
    for i in range(age_max):
        AF[t,i] = n_weight[i]*wealth_F[t,i]
        CF[t,i] = n_weight[i]*consumption_F[t,i]
        CE[t,i] = e_weight[i]*consumption_E[t,i]

    AF_t[t] = AF[t,:].sum() # aggregate capital in the E sector
    CF_t[t] = CF[t,:].sum() # aggregate consumption in the F sector
    CE_t[t] = CE[t,:].sum() # aggregate consumption in the E sector
    
    # the F sector
    if NE_t[t] < N_t[t]:
        KF_t[t] = (alp/(r/(1-ice_t[t])+delta))**(1/(1-alp))*(N_t[t]-NE_t[t]) # aggregate capital in the F sector
        YF_t[t] = KF_t[t]**alp*(N_t[t]-NE_t[t])**(1-alp) # aggregate output in the F sector
        NF_t[t] = N_t[t]-NE_t[t] # aggregate workers in the F sector
    else:
        KF_t[t] = 0.0
        YF_t[t] = 0.0
        NF_t[t] = 0.0


# aggregation
Y_t = YF_t+YE_t
K_t = KF_t+KE_t
C_t = CF_t+CE_t

for t in range(time_max-1):
    
    # private employment share
    NE_N_t[t] = NE_t[t]/N_t[t]
    
    # computing investment in the F sector
    IF_t[t] = (1+g_t)*(1+g_n)*KF_t[t+1]-(1-delta)*KF_t[t]
    
    # computing investment in the E sector
    IE_t[t] = (1+g_t)*(1+g_n)*KE_t[t+1]-(1-delta)*KE_t[t]
    
    # investment rates in the two sectors
    if YF_t[t] > 0:
        IF_Y_t[t] = IF_t[t]/YF_t[t]
    else:
        IF_Y_t[t] = 0.0
    IE_Y_t[t] = IE_t[t]/YE_t[t]
    
    # computing workers' savings
    SF_t[t] = (1+g_t)*(1+g_n)*AF_t[t+1]-AF_t[t]+delta*KF_t[t]
    if YF_t[t] > 0:
        SF_YF_t[t] = SF_t[t]/YF_t[t]

    # computing enterpreneurs' savings
    SE_t[t] = (1+g_t)*(1+g_n)*AE_t[t+1]-AE_t[t]+delta*KE_t[t]
    SE_YE_t[t] = SE_t[t]/YE_t[t]
    
    # aggregate output per capita
    Y_N_t[t] = Y_t[t]/N_t[t]
    
    # aggregate investment rate
    I_Y_t[t] = (IF_t[t]+IE_t[t])/Y_t[t]
    
    # aggregate saving rate
    S_Y_t[t] = (SF_t[t]+SE_t[t])/Y_t[t]

    # capital output ratio
    K_Y_t[t] = K_t[t]/Y_t[t]
    
    # capital outflows
    FA_Y_t[t] = (AE_t[t]+AF_t[t]-K_t[t])/Y_t[t] # stock
    BoP_Y_t[t] = S_Y_t[t]-I_Y_t[t] # flow
    
    if t > 0:
        TFP_t[t] = Y_t[t]/Y_t[t-1]-alp*K_t[t]/K_t[t-1]-(1-alp)*N_t[t]/N_t[t-1]
        YG_t[t] = (Y_t[t]/Y_t[t-1]-1)+g_n+g_t

# figures

time_begin = 0
time_end = 100 #; time_max-1;
tt = [time_begin,time_end]

data_sav=[0.375905127,
0.407118937,
0.417687893,
0.418696583,
0.40780248,
0.410464312,
0.403822419,
0.38944417,
0.377046856,
0.386282215,
0.404312245,
0.432183421,
0.45699599,
0.48157501,
0.501039245,
0.51206739]

data_inv=[0.365907013,
0.425514577,
0.405060796,
0.402900174,
0.38812706,
0.366991801,
0.361881671,
0.361607682,
0.352842054,
0.36494929,
0.378603128,
0.410289533,
0.431546215,
0.427396271,
0.425903209,
0.423250045]

data_res=[0.038897003,
0.033068468,
0.088594251,
0.09722219,
0.117766451,
0.1420134,
0.138692692,
0.140515342,
0.138805234,
0.161149952,
0.196974228,
0.244702191,
0.314965846,
0.355479964,
0.383515959,
0.441448679]

data_em_sh=[0.041140261,
0.063212681,
0.10366673,
0.168350106,
0.232185343,
0.322086332,
0.434391151,
0.474376982,
0.522120471,
0.563805401]

data_SI_Y=[0.009998114,
-0.01839564,
0.012627097,
0.015796409,
0.01967542,
0.043472511,
0.041940748,
0.027836488,
0.024204802,
0.021332925,
0.025709117,
0.021893888,
0.025449774,
0.054178739,
0.075136036,
0.088817345]

# end of year
end_year = 2012

r_F = r/(1-ice_t)

# Panel 1
# Data for plotting
t = np.arange(1992, 2013, 1)
s = r_F[:21]

fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='year',
       title='Panel 1: rate of return in F firms')

ax.set_xlim(1992, 2012)
ax.grid()
#fig.savefig("test.png")
plt.show()

# Panel 2
# Data for plotting
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = NE_N_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1998, 2008, 1)
s = data_em_sh
ax.plot(t, s, label='firm data')
#t = np.arange(1992, 2008, 1)
#s = data_em_sh_agg
#ax.plot(t, s, label='aggregate data')

ax.set(xlabel='year',
       title='Panel 2: E firm employment share')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
#fig.savefig("test.png")
plt.show()

# Panel 3
# Data for plotting
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = S_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = data_sav
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 3: aggregate saving rate')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
#fig.savefig("test.png")
plt.show()

# Panel 4
# Data for plotting
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = I_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = data_inv
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 4: aggregate investment rate')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
#fig.savefig("test.png")
plt.show()

# Panel 5
# Data for plotting
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = FA_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = data_res
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 5: foreign reserves / GDP')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
#fig.savefig("test.png")
plt.show()

# Panel 6
# Data for plotting
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = TFP_t[:len(t)]+(1-alp)*g_t
ax.plot(t, s, label='model')
#t = np.arange(1992, 2008, 1)
#s = data_res
#ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 6: TFP growth rate')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
#fig.savefig("test.png")
plt.show()

# Panel 5
# Data for plotting
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = BoP_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = data_SI_Y
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 7: net export GDP ratio')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
#fig.savefig("test.png")
plt.show()
