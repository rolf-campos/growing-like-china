# -*- coding: utf-8 -*-
"""
   Replication of Zheng Song, Kjetil Storesletten & Fabrizio Zilibotti, 2011.
  "Growing Like China," American Economic Review,  vol. 101(1), pages 196-233.

                                Tests

                         Rodolfo G. Campos
                           January, 2020
"""
import os
import pandas as pd
from replication import Parameters, GLCModel
from scipy.io import loadmat


def test1():
    """
    Test for the initial wealth distribution
    """
    params = Parameters()
    model = GLCModel(params)
    wealth_pre_new = params.initial_ratio * model.life_cycle_profile_pre()
    wealth_pre_E_new = params.initial_ratio_E * model.life_cycle_profile_pre()

    # Compare with Matlab data
    data_pre = loadmat(os.path.join('original_files', 'data_pre.mat'))
    wealth_pre_2 = data_pre['wealth_pre'].flatten()
    check_1 = pd.DataFrame(
            {'Python new': wealth_pre_new, 'Matlab': wealth_pre_2})
    check_1.plot(title='wealth_pre', style=['y-', 'k--'])

    # Compare with Matlab data
    data_pre_E = loadmat(os.path.join('original_files', 'data_pre_E.mat'))
    wealth_pre_E_2 = data_pre_E['wealth_pre_E'].flatten()
    check_2 = pd.DataFrame(
            {'Python new': wealth_pre_E_new, 'Matlab': wealth_pre_E_2})
    check_2.plot(title='wealth_pre_E', style=['y-', 'k--'])


def test2():
    """
    Test for the model solution
    """
    params = Parameters()
    model = GLCModel(params)
    model.solve()

    w_t = model.var.w_t
    m_t = model.var.m_t
    rho_t = model.var.rho_t

    # Compare to the original results
    initial_guess = loadmat(os.path.join('original_files', 'data_result.mat'))

    w_t_orig = initial_guess['w_t'].flatten()
    m_t_orig = initial_guess['m_t'].flatten()
    rho_t_orig = initial_guess['rho_t'].flatten()

    check_1 = pd.DataFrame(
             {'Python new': rho_t, 'Matlab': rho_t_orig})
    check_1.plot(title='rho_t', style=['y-', 'k--'])

    check_2 = pd.DataFrame(
             {'Python new': w_t, 'Matlab': w_t_orig})
    check_2.plot(title='w_t', style=['y-', 'k--'])

    check_3 = pd.DataFrame(
             {'Python new': m_t, 'Matlab': m_t_orig})
    check_3.plot(title='m_t', style=['y-', 'k--'])


if __name__ == "__main__":
    test1()
    test2()
