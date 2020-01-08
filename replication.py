# -*- coding: utf-8 -*-
"""
   Replication of Zheng Song, Kjetil Storesletten & Fabrizio Zilibotti, 2011.
  "Growing Like China," American Economic Review,  vol. 101(1), pages 196-233.

                         Rodolfo G. Campos
                           January, 2020
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pprint
from abc import ABC, abstractmethod
from data import DATA_sav, DATA_inv, DATA_res, DATA_em_sh, DATA_SI_Y
from scipy.io import loadmat


# Utility functions
def pv(stream, r_t, return_flows=False):
    """
    Calculates the present value of any <stream> of flows
    using the discount rate <r_t> (works for r_t scalar or size of <stream>)
    """
    stream = np.asarray(stream)
    r_t = np.asarray(r_t)
    T = stream.size
    if r_t.size == 1 and r_t.size < T:
        r = r_t.sum()
        r_t = np.array(T*[r])
    assert stream.size == r_t.size, "stream = {}, r_t = {}".format(
            stream.size,
            r_t.size)
    # factor_t = np.array([(1.+r_t)**(-t) for t in range(T)])
    tt = np.array([-t for t in range(T)])
    factor_t = (np.ones(T) + r_t)**tt
    factor_t = np.ones(T) / (np.ones(T) + r_t)
    factor_t[0] = 1.
    flows = stream * factor_t.cumprod()
    if return_flows:
        return flows.sum(), flows
    else:
        return flows.sum()


def transition_vector(length, v0, v1, t0, t1, speed=1):
    """
    Returns a vector of size <length> that transitions from <v0> to <v1>
    starting at date <t0> and completing the transition at <t1>.
    Speed controls the speed; <speed> = 1 is linear.
    Vector is zero-indexed, so position t corresponds to date t+1.
    """
    assert t0 < t1, "t0={} < t1={}".format(t0, t1)
    assert t1 <= length, "t1={} <= length={}".format(t1, length)
    T = length
    vector = np.empty(T)
    vector[:t0] = v0
    vector[t1-1:] = v1
    for t in range(t0, t1):
        weight = ((t-t0) / (t1-t0))**speed
        vector[t] = (1.0-weight)*v0 + weight*v1
    return vector


def steps(d, size):
    """
    Produces a np.array of length <size> that increases in steps
    A dictionary
        d = {idx_1: val_1, idx_2: val_2,...}
    produces
        np.array([0, 0, val_1,..., val_1, val_2,..., val_2,...])
    where the switches occur at idx_1, idx_2,...
    Requirement 0 <= idx_j < size, for all j is enforced
    """
    a = np.zeros(size)
    if d:  # an empty dictionary is allowed
        locations = sorted(list(d.keys()))
        assert locations[0] >= 0, "An index in dictionary is out of bounds"
        assert locations[-1] < size, "An index in dictionary is out of bounds"
        for i in locations:
            a[i:] = d[i]
    return a


class Parameters:
    """
    Class to encapsulate all parameters of the model.
    Default values are those of the original "Growing like China" paper.
    """
    def __init__(self,
                 name=None,  # name of the set of parameters (e.g., "Baseline")
                 beta=0.998,  # discount factor of workers
                 beta_E=0.998,  # discount factor of enterpreneurs
                 r=0.0175,  # world interest rate
                 sigma=0.5,  # the inverse of intertemporal substitution
                 alpha=0.5,  # capital output elasticity
                 delta=0.1,  # depreciation rate
                 g_n=0.03,  # exogenous population growth
                 r_soe_ini=0.093,  # initial lending rate for SOEs
                 g_t=0.038,  # exogenous TFP growth
                 KY_F_E=2.65,  # ratio of K/Y in the F sector vs the E sector
                 loan_asset=1.0,   # loan asset ratio in the E sector
                 # initial assets
                 initial_ratio=0.80,
                 initial_ratio_E=0.33,
                 # demographic structure
                 age_max=50,  # maximum age
                 age_T=26,  # the age when entrepreneurs become firm owners
                 age_T_w=31,  # the age when workers retire
                 time_max=400,  # the end of the economy
                 n_pre=100,  # the initial size of workers
                 e_pre=5  # the initial size of entrepreneurs
                 ):
        """
        Initialize parameters and also compute some derived ones
        """
        # Hack to easily load attributes with default values:
        args = locals().copy()
        for key, val in args.items():
            if key == "self":
                pass
            else:
                setattr(self, key, val)

        # Set attributes that require some computation:
        self.set_derived_parameters()

    def __repr__(self):
        d = self.__dict__.copy()
        name = d.pop("name")
        if name:
            header = "Parameters <{}>:\n".format(name)
        else:
            header = "Parameters: \n"
        pp = pprint.PrettyPrinter(indent=1)
        s = header + pp.pformat(d)
        return s

    def to_dictionary(self):
        """
        Return copy of parameters in dictionary format
        """
        return self.__dict__.copy()

    def set_derived_parameters(self):
        """
        Compute the derived parameters
        """
        r, alpha, delta = self.r, self.alpha, self.delta
        r_soe_ini, KY_F_E = self.r_soe_ini, self.KY_F_E
        loan_asset = self.loan_asset

        # iceberg cost
        iceberg = 1.0 - self.r / (self.r_soe_ini)
        self.iceberg = iceberg

        # T = self.time_max + self.age_max - 1
        self.ice_t = self.financial_reform(start=9, end=27, speed=2.38)

        # ratio of the rate of return in the E sector to that in the F sector
        rho_r = (r_soe_ini+0.09) / (r/(1.0-iceberg))
        self.rho_r = rho_r

        # share of managerial compensation
        self.psi = (
                1. - (rho_r*r/(1.-iceberg)+delta)
                / (r/(1.-iceberg)+delta)
                / KY_F_E
                )

        # productivity ratio of E over F
        self.ksi = (
                (KY_F_E)**(alpha/(1.-alpha)) / (1.-self.psi)
                )

        # measure of financial frictions
        # Formula: eta = loan_asset * (1+r/(1-ice))
        # / (1+rho_r*r/(1-ice)+(rho_r*r/(1-ice)-r/(1-ice))*loan_asset)
        t0 = 1. - iceberg
        t1 = r / t0
        t2 = rho_r * t1
        self.eta = loan_asset * (1+t1) / (1.+t2+(t2-t1)*loan_asset)

        # pre-transition wage
        self.w_pre = (
                (1.-alpha) * (alpha/(r/(1.-iceberg)+delta))**(alpha/(1.-alpha))
                )

        # Check whether Assumption 1 (on p. 210) holds
        value_a1 = self.ksi - (1./(1.-self.psi))**(1./(1.-alpha))
        if value_a1 < 0:
            print('Assumption 1: > 0 is false')
            print(value_a1)

    def financial_reform(self, start=9, end=27, speed=2.38):
        """
        Capital deepening (reform in the financial sector)

        Returns a vector of size T containing variable iceberg costs.
        The cost is set to <iceberg> for t <= <time_start> and to zero
        for t>= <time end>.
        T should be time_max + age_max - 1
        Original paper uses speed = 2.38
        """
        T = self.time_max + self.age_max - 1
        return transition_vector(T,
                                 self.iceberg,
                                 0.0,
                                 start-1,
                                 end-1,
                                 speed
                                 )

    def adjust_rho(self, rho_t):
        """
        Adjustment of the rate of return
        due to the endogenous borrowing constraint
        """
        r = self.r
        eta = self.eta
        ice_t = self.ice_t
        c_t = (  # TBD: verify formula
              (rho_t*(1.+r/(1.-ice_t))+eta*(rho_t-r/(1.-ice_t)))
              / (1.+r/(1.-ice_t)-eta*(rho_t-r/(1.-ice_t)))
              )
        return np.maximum(rho_t, c_t)

    def demographic_distribution(self):
        """
        Calculate the distribution of the population according to age
        given the fertility rate g_n and age_max.
        Returns a np.array of size <age_max>.
        """
        g_n = self.g_n
        age_max = self.age_max
        if float(g_n) == 0.0:
            # Uniform across age
            profile = 1. / age_max * np.ones(age_max)
        else:
            population = np.empty(age_max)
            for i in range(age_max):
                age = i + 1
                population[i] = (1.+g_n)**(age_max-age)
            total_population = (1.-(1.+g_n)**age_max) / (1.-(1.+g_n))
            profile = population / total_population
        return profile


class Agent(ABC):
    """
    Class common to all agents in the model.
    Subclassed by the Worker and Entrepreneur classes.
    """
    def __init__(self,
                 name,  # name of the agent (e.g., "Worker")
                 age,  # current age of the agent
                 age_max,  # maximum age
                 beta,  # discount factor
                 sigma,  # the inverse of intertemporal substitution
                 ):
        """
        Initialize parameters
        """
        args = locals().copy()
        for key, val in args.items():
            if key == "self":
                pass
            else:
                setattr(self, key, val)

    def __repr__(self):
        d = self.__dict__.copy()
        name = d.pop("name")
        if name:
            header = "Agent <{}>:\n".format(name)
        else:
            header = "Agent: \n"
        pp = pprint.PrettyPrinter(indent=1)
        s = header + pp.pformat(d)
        return s

    def to_dictionary(self):
        """
        Return copy of parameters in dictionary format
        """
        return self.__dict__.copy()

    def income_steps(self, income_by_age, g=False):
        """
        Generate a generic income stream (np.array) of length <age_max> of
        the form a = [0,...0, inc1,..., inc1..., inc2,... inc2,...],
        where the inc1 starts at age1, inc2 at age2, etc.

        If g != 0, non-zero values are multiplied by (1+g)^jm j = 0, 1, 2,...

        The counter starts at j = 0 at age_1.
        """
        size = self.age_max
        d = {key-1: val for key, val in income_by_age.items()}
        a = steps(d, size)
        if g:
            sorted_ages = sorted(list(income_by_age.keys()))
            first_age = sorted_ages[0]
            life_span = self.age_max - first_age + 1
            growth = np.array([(1.+g)**j for j in range(life_span)])
            a[-life_span:] = a[-life_span:] * growth
        return a

    def euler(self, r_t, g=False):
        """
        Generates the factors from the Euler equation for CRRA utility

        Size is inferred from the size of <r_t>.

        If g != 0, then g is used to obtain detrended consumption
        """
        r_t = np.asarray(r_t)
        factor_t = (self.beta*(1.+r_t))**(1./self.sigma)
        if g:
            g = np.asarray(g)
            assert g.size == 1 or g.size == r_t.size
            factor_t = factor_t / (1.+g)
        factor_t[0] = 1.  # normalize
        return factor_t

    @abstractmethod
    def income(self):
        """
        To be implemented by subclasses
        """
        pass

    @abstractmethod
    def optimize(self):
        """
        To be implemented by subclasses
        """
        pass


class Worker(Agent):
    """
    Class for workers
    """
    def __init__(self,
                 name=None,  # name of the agent (e.g., "Worker")
                 age=1,  # age of the agent
                 age_max=50,  # maximum age
                 beta=0.998,  # discount factor
                 sigma=0.5,  # the inverse of intertemporal substitution
                 job='Worker',
                 age_retire=31,  # the age when workers retire
                 wealth=0.,  # wealth at current age
                 wage=None,
                 pension=0.,
                 year=1
                 ):
        super().__init__(name,
                         age,
                         age_max,
                         beta,
                         sigma)
        self.job = job
        self.age_retire = age_retire
        self.wealth = wealth
        self.wage = wage
        self.pension = pension
        self.year = year
        self.income_by_age = {
                self.age: self.wage,
                self.age_retire: self.pension
                }

    def income0(self, g=False):
        """
        Computes income stream of a worker given all attributes.
        Sets the income_stream = np.array of size <age_max>.
        """
        income_stream = self.income_steps(
                self.income_by_age,
                g
                )
        income_stream[:self.age-1] = 0.  # Erase income in the past
        return income_stream

    def optimize0(self, environment):
        """
        Saving, wealth_prime, saving rate, consumption for workers
        """
        # Environment variables
        r = environment.r
        g_t = environment.g_t
        # Agent variables
        age_max = self.age_max
        age = self.age
        beta = self.beta
        sigma = self.sigma
        wealth = self.wealth
        income_stream = self.income0(g=environment.g_t)
        wage = income_stream[age-1]

        # optimal consumption and savings
        A = pv(income_stream[self.age-1:], r) + wealth*(1.+r)

        life_span = age_max - age + 1  # includes current age
        # factor^t for each c_t when Euler eq. is substituted
        # in budget constraint
        factor = (beta*(1.+r))**(1./sigma)/(1.+r)
        ratio = [factor**j for j in range(life_span)]
        ratio = np.array(ratio)
        consumption = A / (np.sum(ratio))
        saving = wealth*r + wage - consumption
        sr = saving / (wealth*r+wage)  # saving rate

        # computing next-period wealth
        wealth_prime = wealth*(1.+r) + wage - consumption
        # adjustment because of detrending
        wealth_prime_detrended = wealth_prime / (1.+g_t)
        return wealth_prime_detrended, sr, consumption

    def income(self, w_t, g=False):
        """
        Computes income stream of a worker given all attributes.
        Returns np.array of size <age_max>.
        """
        relevant_w_t = w_t[self.year-1:]

        income_by_age = dict()
        for i, age in enumerate(range(self.age, self.age_retire)):
            income_by_age[age] = relevant_w_t[i]
        income_by_age[self.age_retire] = self.pension

        income_stream = self.income_steps(
                income_by_age,
                g
                )
        # income_stream[:self.age-1] = 0.  # Erase income in the past
        self.ii = income_stream
        return income_stream

    def optimize(self, environment, w_t, m_t, r_t):
        """
        Optimal decisions for consumption and wealth
        """
        r_t = environment.r * np.ones(self.age_max-self.age+1)

        W = self.wealth * (1.+r_t[0])
        income = self.income(w_t, g=environment.g_t)
        income = income[self.age-1:]  # get rid of past
        pv_income = pv(income, environment.r)
        wealth_0 = pv_income + W

        euler = self.euler(r_t, g=False)
        ratio = pv(euler.cumprod(), r_t)
        c_0 = wealth_0 / ratio.sum()

        euler_detrended = self.euler(r_t, g=environment.g_t)
        cons = c_0 * euler_detrended.cumprod()

        income_detrended = self.income(w_t, g=False)
        income_detrended = income_detrended[self.age-1:]  # get rid of past

        w = [self.wealth]
        for i, c in enumerate(cons[:-1]):
            w_income = w[i] * (1.+r_t[i])
            w_prime = w_income + income_detrended[i] - c
            w.append(w_prime/(1.+environment.g_t))
        w = np.array(w)

        return {'consumption': cons, 'wealth': w}


class Entrepreneur(Agent):
    """
    Class for entrepreneurs
    """
    def __init__(self,
                 name=None,  # name of the agent (e.g., "Entrepreneur")
                 age=1,  # age of the agent
                 age_max=50,  # maximum age
                 beta=0.998,  # discount factor
                 sigma=0.5,  # the inverse of intertemporal substitution
                 job='Entrepreneur',
                 age_T=26,  # the age when entrepreneurs become firm owners
                 wealth=0.,  # wealth at current age
                 # income_by_age=dict(),
                 year=1
                 ):
        super().__init__(name,
                         age,
                         age_max,
                         beta,
                         sigma)
        self.job = job
        self.age_T = age_T
        self.wealth = wealth
        # self.income_by_age = dict(income_by_age)
        self.year = year

    def income(self, w_t, g=False):
        """
        Computes income stream of an entrepreneur given all attributes.
        Returns np.array of size <age_max>.
        """
        relevant_w_t = w_t[self.year-1:]

        income_by_age = dict()
        for i, age in enumerate(range(self.age, self.age_T)):
            income_by_age[age] = relevant_w_t[i]
        income_by_age[self.age_T] = 0.

        income_stream = self.income_steps(
                income_by_age,
                g
                )
        # income_stream[:self.age-1] = 0.  # Erase income in the past
        return income_stream

    def optimize(self, environment, w_t, m_t, r_t):
        """
        Optimal decisions for consumption and wealth
        """
        r_t = environment.adjust_rho(r_t)
        r_t = r_t[self.year-1:]  # relevant part of r_t
        if self.age < self.age_T:
            periods_manager = self.age_T-self.age
            r_t[:periods_manager] = environment.r

        r_t = r_t[:self.age_max-self.age+1]  # remaining life-span

        W = self.wealth * (1.+r_t[0])
        income = self.income(m_t, g=environment.g_t)
        income = income[self.age-1:]  # get rid of past
        pv_income = pv(income, environment.r)
        wealth_0 = pv_income + W

        euler = self.euler(r_t, g=False)
        ratio = pv(euler.cumprod(), r_t)
        c_0 = wealth_0 / ratio.sum()

        euler_detrended = self.euler(r_t, g=environment.g_t)
        cons = c_0 * euler_detrended.cumprod()

        income_detrended = self.income(m_t, g=False)
        income_detrended = income_detrended[self.age-1:]  # get rid of past

        w = [self.wealth]
        for i, c in enumerate(cons[:-1]):
            w_income = w[i] * (1.+r_t[i])
            w_prime = w_income + income_detrended[i] - c
            w.append(w_prime/(1.+environment.g_t))
        w = np.array(w)

        return {'consumption': cons, 'wealth': w}


# Main functions
def life_cycle_profile_pre(environment):
    # Dummy worker to obtain age_max and a vector of wages
    w1 = Worker(age=1, wage=environment.w_pre)
    age_max = w1.age_max
    wages = w1.income0(g=0.)

    wealth_pre = np.zeros(age_max)
    consumption_pre = np.zeros(age_max)
    sr_pre = np.zeros(age_max)
    for i in range(age_max):
        age = i + 1
        wage = wages[i]
        wealth = wealth_pre[i]
        w = Worker(age=age, wage=wage, wealth=wealth)
        wealth_next, sr_pre[i], consumption_pre[i] = w.optimize0(environment)
        if age < age_max:
            wealth_pre[i+1] = wealth_next
    return wealth_pre


params = Parameters()
wealth_pre_new = params.initial_ratio * life_cycle_profile_pre(params)
wealth_pre_E_new = params.initial_ratio_E * life_cycle_profile_pre(params)

# Compare with Matlab data
data_pre = loadmat(os.path.join('original_files', 'data_pre.mat'))
wealth_pre_2 = data_pre['wealth_pre'].flatten()
check_1 = pd.DataFrame(
        {'Python new': wealth_pre_new, 'Matlab': wealth_pre_2})
check_1.plot(title='wealth_pre', style=['y-', 'k--'])

# Compare with Matlab data
data_pre_E = loadmat(os.path.join('original_files', 'data_pre_E.mat'))
wealth_pre_E_2 = data_pre_E['wealth_pre_E'].flatten()
check_1 = pd.DataFrame(
        {'Python new': wealth_pre_E_new, 'Matlab': wealth_pre_E_2})
check_1.plot(title='wealth_pre_E', style=['y-', 'k--'])

params = Parameters()

pop_weight = params.demographic_distribution()
# the initial size of workers before retirement
nw_pre = np.sum(pop_weight[0:params.age_T_w-1]) * params.n_pre
# the initial size of entrepreneurs after being firm owner
ee_pre = np.sum(pop_weight[params.age_T-1:params.age_max]) * params.e_pre

# iteration choices
relax = 0.75
iter_max = 1000
tol = 1e-4
dev_max = 1.
iteration = 0

# initial guess = true results
initial_guess = loadmat(os.path.join('original_files', 'data_result.mat'))

w_t = initial_guess['w_t'].flatten()
m_t = initial_guess['m_t'].flatten()
rho_t = initial_guess['rho_t'].flatten()

# Parameters
r = params.r
eta = params.eta
ice_t = params.ice_t
alpha = params.alpha
ksi = params.ksi
psi = params.psi
delta = params.delta
g_n = params.g_n
g_t = params.g_t

while dev_max > tol and iteration < iter_max:
    # an indicator for the end of transition
    I_end = 0

    # Initialize all vectors and matrices
    SHAPE_LONG_WIDE = (params.time_max+params.age_max-1, params.age_max)
    SHAPE_SHORT_WIDE = (params.time_max, params.age_max)
    SHAPE_SHORT = params.time_max
    SHAPE_LONG = params.time_max + params.age_max - 1

    ae = np.zeros(SHAPE_SHORT_WIDE)
    AE = np.zeros(SHAPE_SHORT_WIDE)
    loan = np.zeros(SHAPE_SHORT_WIDE)
    ke = np.zeros(SHAPE_SHORT_WIDE)
    ne = np.zeros(SHAPE_SHORT_WIDE)
    KE = np.zeros(SHAPE_SHORT_WIDE)
    NE = np.zeros(SHAPE_SHORT_WIDE)
    LE = np.zeros(SHAPE_SHORT_WIDE)

    wealth_E = np.zeros(SHAPE_LONG_WIDE)
    consumption_E = np.zeros(SHAPE_LONG_WIDE)

    AE_t = np.zeros(SHAPE_SHORT)
    NE_t = np.zeros(SHAPE_SHORT)
    KE_t = np.zeros(SHAPE_SHORT)
    LE_t = np.zeros(SHAPE_SHORT)
    N_t = np.zeros(SHAPE_SHORT)
    YE_t = np.zeros(SHAPE_SHORT)
    M_t = np.zeros(SHAPE_SHORT)
    loan_ratio = np.zeros(SHAPE_SHORT)

    w_t_new = np.zeros(SHAPE_LONG)
    rho_t_new = np.zeros(SHAPE_LONG)
    m_t_new = np.zeros(SHAPE_LONG)

    # existing entrepreneurs
    # includes entrepreneurs aged age_max but not aged one (they would be new)
    for age in range(2, params.age_max+1):
        ii = age - 1
        # computing existing entrepreneurs' wealth
        # given the guess of m_t and rho_t
        e = Entrepreneur(age=age, wealth=wealth_pre_E_new[ii])
        result = e.optimize(params, w_t, m_t, rho_t)
        consumption, wealth = result['consumption'], result['wealth']
        # wealth and cons time series for the existing enterpreneurs
        for tt in range(params.age_max-ii):
            wealth_E[tt, ii+tt] = wealth[tt]
            consumption_E[tt, ii+tt] = consumption[tt]

    # newly-born entrepreneurs
    for tt in range(params.time_max):
        year = tt+1
        e = Entrepreneur(age=1, wealth=0., year=year)
        result = e.optimize(params, w_t, m_t, rho_t)
        consumption, wealth = result['consumption'], result['wealth']
        # wealth and cons time series for the existing enterpreneurs
        for ii in range(params.age_max):
            wealth_E[tt+ii, ii] = wealth[ii]
            consumption_E[tt+ii, ii] = consumption[ii]

    # Update new factor price time series
    for t in range(params.time_max):

        # Fixed size of managers
        E_t = params.e_pre - ee_pre

        # Assets in the E sector
        for i in range(params.age_max):
            # entrepreneurial capital owned by an entprepreneur at time t
            # with age i
            ae[t, i] = wealth_E[t, i]
            # total capital owned by all entrepreneurs at time t with age i
            AE[t, i] = params.e_pre * pop_weight[i] * ae[t, i]

        # capital and labor in the E sector
        for i in range(params.age_T-1, params.age_max):
            if rho_t[t] >= r / (1.-ice_t[t]):  # borrowing is profitable
                loan_ratio[t] = (
                        eta * (1.+rho_t[t])
                        / (1.+r/(1.-ice_t[t])-eta*(rho_t[t]-r/(1.-ice_t[t])))
                        )  # loan asset ratio
                loan[t, i] = wealth_E[t, i] * loan_ratio[t]
                # entrepreneurial capital owned by an entrepreneur at time t
                # with age i
                ke[t, i] = wealth_E[t, i] + loan[t, i]
            else:  # borrowing is not profitable
                loan[t, i] = 0.
                # entrepreneurial capital owned by an entrepreneur at time t
                # with age i
                ke[t, i] = wealth_E(t, i)

            # labor employed by an entrepreneur at time with age i
            ne[t, i] = ke[t, i] * (
                    (1.-alpha)*(1.-psi)*ksi**(1.-alpha)/w_t[t])**(1./alpha)
            # total capital owned by all entrepreneurs at time with age i
            KE[t, i] = params.e_pre * pop_weight[i] * ke[t, i]
            # total labor employed by all entrepreneurs at time with age i
            NE[t, i] = params.e_pre * pop_weight[i] * ne[t, i]
            # total loan
            LE[t, i] = params.e_pre * pop_weight[i] * loan[t, i]

        # resource allocation
        AE_t[t] = AE[t, :].sum()  # aggregate capital in the E sector
        NE_t[t] = NE[t, :].sum()  # aggregate employment in the E sector
        KE_t[t] = KE[t, :].sum()  # when rho > r
        LE_t[t] = LE[t, :].sum()  # total loan
        N_t[t] = nw_pre  # the size of workers (no migration)

        # factor prices
        # wage rate
        w_t_new[t] = (
                (1.-psi) * (1.-alpha) * (KE_t[t]/NE_t[t])**alpha
                * ksi**(1.-alpha)
                )

        # locate the end of the transition
        if NE_t[t] >= N_t[t] and I_end == 0:
            I_end = 1
            I_t = t
        elif I_end == 1:
            I_end = 1

        if I_end == 0:
            w_t_new[t] = (
                    (1.-alpha)
                    * (alpha/(r/(1.-ice_t[t])+delta))**(alpha/(1.-alpha))
                    )  # wage rate
        else:
            NE_t[t] = N_t[t]
            w_t_new[t] = (
                    (1.-psi) * (1.-alpha) * (KE_t[t]/N_t[t])**alpha
                    * ksi**(1.-alpha)
                    )  # wage rate

        # the internal rate of return for entrepreneurs
        rho_t_new[t] = np.max(
                [r,
                 (
                         (1.-psi)**(1./alpha) * ksi**((1.-alpha)/alpha)
                         * ((1.-alpha)/w_t_new[t])**((1.-alpha)/alpha) * alpha
                         - delta
                 )
                 ]
                )
        # aggregate output in the E sector
        YE_t[t] = KE_t[t]**alpha * (ksi*NE_t[t])**(1.-alpha)
        # total managerial compensations
        M_t[t] = psi * YE_t[t]
        # compensations for young entrepreneurs
        m_t_new[t] = M_t[t] / E_t

    # steady state assumption
    w_t_new[params.time_max:] = w_t_new[params.time_max-1]
    rho_t_new[params.time_max:] = rho_t_new[params.time_max-1]
    m_t_new[params.time_max:] = m_t_new[params.time_max-1]

    # deviation
    dev_w = np.abs(w_t_new-w_t)
    dev_rho = np.abs(rho_t_new-rho_t)
    dev_m = np.abs(m_t_new-m_t)
    dev_w_max = dev_w.max()
    dev_rho_max = dev_rho.max()
    dev_m_max = dev_m.max()
    dev_max = np.array([dev_w_max, dev_rho_max, dev_m_max]).max()

    # renew
    w_t = w_t*relax + w_t_new*(1.-relax)
    rho_t = rho_t*relax + rho_t_new*(1.-relax)
    m_t = m_t*relax + m_t_new*(1.-relax)

    if int(5*np.floor(iteration/5)) == iteration:
        print("Iteration: {0}, Max deviation: {1}".format(iteration, dev_max))

    iteration += 1

# Compare to the original results
w_t_orig = initial_guess['w_t'].flatten()
m_t_orig = initial_guess['m_t'].flatten()
rho_t_orig = initial_guess['rho_t'].flatten()

check = pd.DataFrame(
        {'Python new': rho_t, 'Matlab': rho_t_orig})
check.plot(title='rho_t', style=['y-', 'k--'])

check = pd.DataFrame(
        {'Python new': w_t, 'Matlab': w_t_orig})
check.plot(title='w_t', style=['y-', 'k--'])

check = pd.DataFrame(
        {'Python new': m_t, 'Matlab': m_t_orig})
check.plot(title='m_t', style=['y-', 'k--'])

# Initialize all vectors and matrices
wealth_F = np.zeros(SHAPE_LONG_WIDE)
consumption_F = np.zeros(SHAPE_LONG_WIDE)
N_t = np.zeros(SHAPE_SHORT)
AF = np.zeros(SHAPE_SHORT_WIDE)
CF = np.zeros(SHAPE_SHORT_WIDE)
CE = np.zeros(SHAPE_SHORT_WIDE)
AF_t = np.zeros(SHAPE_SHORT)
CF_t = np.zeros(SHAPE_SHORT)
CE_t = np.zeros(SHAPE_SHORT)
KF_t = np.zeros(SHAPE_SHORT)
YF_t = np.zeros(SHAPE_SHORT)
NF_t = np.zeros(SHAPE_SHORT)
NE_N_t = np.zeros(SHAPE_SHORT)
IF_t = np.zeros(SHAPE_SHORT)
IE_t = np.zeros(SHAPE_SHORT)
IF_Y_t = np.zeros(SHAPE_SHORT)
IE_Y_t = np.zeros(SHAPE_SHORT)
SF_t = np.zeros(SHAPE_SHORT)
SF_YF_t = np.zeros(SHAPE_SHORT)
SE_t = np.zeros(SHAPE_SHORT)
SE_YE_t = np.zeros(SHAPE_SHORT)
Y_N_t = np.zeros(SHAPE_SHORT)
I_Y_t = np.zeros(SHAPE_SHORT)
S_Y_t = np.zeros(SHAPE_SHORT)
K_Y_t = np.zeros(SHAPE_SHORT)
FA_Y_t = np.zeros(SHAPE_SHORT)
BoP_Y_t = np.zeros(SHAPE_SHORT)
TFP_t = np.zeros(SHAPE_SHORT)
YG_t = np.zeros(SHAPE_SHORT)

# workers' savings and assets
for age in range(2, params.age_max+1):
    ii = age - 1
    # computing existing workers' wealth given the guess of m_t and rho_t
    w = Worker(age=age, wealth=wealth_pre_new[ii])
    result = w.optimize(params, w_t, m_t, rho_t)
    consumption, wealth = result['consumption'], result['wealth']
    # wealth and cons time series for the existing workers
    for tt in range(params.age_max-ii):
        wealth_F[tt, ii+tt] = wealth[tt]
        consumption_F[tt, ii+tt] = consumption[tt]

# newly-born workers
for tt in range(params.time_max):
    year = tt + 1
    w = Worker(age=1, wealth=0., year=year)
    result = w.optimize(params, w_t, m_t, rho_t)
    consumption, wealth = result['consumption'], result['wealth']
    for ii in range(params.age_max):
        wealth_F[tt+ii, ii] = wealth[ii]
        consumption_F[tt+ii, ii] = consumption[ii]

# demographic structure and others
for t in range(params.time_max):

    # no migration
    N_t[t] = nw_pre

    # total assets of workers and total consumptions
    for i in range(params.age_max):
        AF[t, i] = params.n_pre * pop_weight[i] * wealth_F[t, i]
        CF[t, i] = params.n_pre * pop_weight[i] * consumption_F[t, i]
        CE[t, i] = params.e_pre * pop_weight[i] * consumption_E[t, i]

    AF_t[t] = AF[t, :].sum()  # aggregate capital in the E sector
    CF_t[t] = CF[t, :].sum()  # aggregate consumption in the F sector
    CE_t[t] = CE[t, :].sum()  # aggregate consumption in the E sector

    # the F sector
    if NE_t[t] < N_t[t]:
        KF_t[t] = (
                (alpha/(r/(1.-ice_t[t])+delta))**(1./(1.-alpha))
                * (N_t[t]-NE_t[t])
                )  # aggregate capital in the F sector
        YF_t[t] = (
                KF_t[t]**alpha * (N_t[t]-NE_t[t])**(1.-alpha)
                )  # aggregate output in the F sector
        NF_t[t] = N_t[t] - NE_t[t]  # aggregate workers in the F sector
    else:
        KF_t[t] = 0.0
        YF_t[t] = 0.0
        NF_t[t] = 0.0


# aggregation
Y_t = YF_t + YE_t
K_t = KF_t + KE_t
C_t = CF_t + CE_t

for t in range(params.time_max-1):

    # private employment share
    NE_N_t[t] = NE_t[t] / N_t[t]

    # computing investment in the F sector
    IF_t[t] = (1.+g_t)*(1.+g_n)*KF_t[t+1] - (1.-delta)*KF_t[t]

    # computing investment in the E sector
    IE_t[t] = (1.+g_t)*(1.+g_n)*KE_t[t+1] - (1.-delta)*KE_t[t]

    # investment rates in the two sectors
    if YF_t[t] > 0:
        IF_Y_t[t] = IF_t[t] / YF_t[t]
    else:
        IF_Y_t[t] = 0.0
    IE_Y_t[t] = IE_t[t] / YE_t[t]

    # computing workers' savings
    SF_t[t] = (1.+g_t)*(1.+g_n)*AF_t[t+1] - AF_t[t] + delta*KF_t[t]
    if YF_t[t] > 0:
        SF_YF_t[t] = SF_t[t] / YF_t[t]

    # computing enterpreneurs' savings
    SE_t[t] = (1.+g_t)*(1.+g_n)*AE_t[t+1] - AE_t[t] + delta*KE_t[t]
    SE_YE_t[t] = SE_t[t] / YE_t[t]

    # aggregate output per capita
    Y_N_t[t] = Y_t[t] / N_t[t]

    # aggregate investment rate
    I_Y_t[t] = (IF_t[t]+IE_t[t]) / Y_t[t]

    # aggregate saving rate
    S_Y_t[t] = (SF_t[t]+SE_t[t]) / Y_t[t]

    # capital output ratio
    K_Y_t[t] = K_t[t] / Y_t[t]

    # capital outflows
    FA_Y_t[t] = (AE_t[t]+AF_t[t]-K_t[t]) / Y_t[t]  # stock
    BoP_Y_t[t] = S_Y_t[t] - I_Y_t[t]  # flow

    if t > 0:
        TFP_t[t] = (
                Y_t[t]/Y_t[t-1]
                - alpha*K_t[t]/K_t[t-1]
                - (1.-alpha)*N_t[t]/N_t[t-1]
                )
        YG_t[t] = (Y_t[t]/Y_t[t-1]-1.) + g_n + g_t

# Figures
time_begin = 0
time_end = 100  # ; time_max-1;
tt = [time_begin, time_end]

# end year
end_year = 2012

# Panel 1
r_F = r / (1.-ice_t)
t = np.arange(1992, 2013, 1)
s = r_F[:21]
fig, ax = plt.subplots()
ax.plot(t, s)
ax.set(xlabel='year',
       title='Panel 1: rate of return in F firms')

ax.set_xlim(1992, 2012)
ax.grid()
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()

# Panel 2
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = NE_N_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1998, 2008, 1)
s = DATA_em_sh
ax.plot(t, s, label='firm data')

ax.set(xlabel='year',
       title='Panel 2: E firm employment share')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()

# Panel 3
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = S_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = DATA_sav
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 3: aggregate saving rate')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()

# Panel 4
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = I_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = DATA_inv
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 4: aggregate investment rate')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()

# Panel 5
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = FA_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = DATA_res
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 5: foreign reserves / GDP')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()

# Panel 6
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = TFP_t[:len(t)]+(1.-params.alpha)*params.g_t
ax.plot(t, s, label='model')

ax.set(xlabel='year',
       title='Panel 6: TFP growth rate')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()

# Panel 5
fig, ax = plt.subplots()
t = np.arange(1992, end_year+1, 1)
s = BoP_Y_t[:len(t)]
ax.plot(t, s, label='model')
t = np.arange(1992, 2008, 1)
s = DATA_SI_Y
ax.plot(t, s, label='data')

ax.set(xlabel='year',
       title='Panel 7: net export GDP ratio')

ax.set_xlim(1992, 2012)
ax.grid()
ax.legend(loc='upper left')
plt.xticks(np.arange(1992, 2013, step=2))
plt.show()
