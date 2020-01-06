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
                 pension=0.
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
        self.income_by_age = {
                self.age: self.wage,
                self.age_retire: self.pension
                }

    def income(self, g=False):
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
        income_stream = self.income(g=environment.g_t)
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

    def optimize(self, environment):
        return self.optimize0(environment)


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
        # print('r_t: {}'.format(r_t))
        W = self.wealth * (1.+r_t[0])
        # print(r_t[0])
        income = self.income(m_t, g=environment.g_t)
        income = income[self.age-1:]  # get rid of past
        # print('income: {}'.format(income))
        pv_income = pv(income, environment.r)
        wealth_0 = pv_income + W
        # print('Wealth: {}'.format(wealth_0))

        # print('r_t: {}'.format(r_t))
        euler = self.euler(r_t, g=False)
        # print('euler: {}'.format(euler))
        ratio, rflows = pv(euler.cumprod(), r_t, True)
        # print('rflows: {}'.format(rflows))
        euler_detrended = self.euler(r_t, g=environment.g_t)
        c_0 = wealth_0 / ratio.sum()
        cons = c_0 * euler_detrended.cumprod()

        income_detrended = self.income(m_t, g=False)
        income_detrended = income_detrended[self.age-1:]  # get rid of past
        # print('income detrended: {}'.format(income_detrended))

        w = [self.wealth]
        for i, c in enumerate(cons[:-1]):
            w_income = w[i] * (1.+r_t[i])
            w_prime = w_income + income_detrended[i] - c
            w.append(w_prime/(1.+environment.g_t))
        w = np.array(w)
        return {'consumption': cons, 'wealth': w}

    def euler(self, r_t, g=False):
        """
        g is used to obtain detrended consumption
        """
        r_t = np.asarray(r_t)
        factor_t = (self.beta*(1.+r_t))**(1./self.sigma)
        if g:
            g = np.asarray(g)
            assert g.size == 1 or g.size == r_t.size
            factor_t = factor_t / (1.+g)
        factor_t[0] = 1.  # normalize
        return factor_t

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


# Main functions
def life_cycle_profile_pre(environment):
    # Dummy worker to obtain age_max and a vector of wages
    w1 = Worker(age=1, wage=environment.w_pre)
    age_max = w1.age_max
    wages = w1.income(g=0.)

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
# initial guess = true results
initial_guess = loadmat(os.path.join('original_files', 'data_result.mat'))

w_t = initial_guess['w_t'].flatten()
m_t = initial_guess['m_t'].flatten()
rho_t = initial_guess['rho_t'].flatten()

e = Entrepreneur(age=1, wealth=0.0, year=100)
tr1 = e.optimize(params, w_t, m_t, rho_t)

# tr3 = saving_E_newly_born(e.year)
# tr3 = saving_E_existing(e.age, e.wealth)
print(e)
print(tr1['consumption'])
# print(tr2[1])
# print(tr3[1])

print(tr1['wealth'])
# print(tr3[0])
