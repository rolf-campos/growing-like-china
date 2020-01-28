# -*- coding: utf-8 -*-
"""
   Replication of Zheng Song, Kjetil Storesletten & Fabrizio Zilibotti, 2011.
  "Growing Like China," American Economic Review,  vol. 101(1), pages 196-233.

                    Helper functions and classes

                         Rodolfo G. Campos
                           January, 2020
"""
import numpy as np
import pandas as pd


class AttributeDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([key.rjust(m) + ': ' + repr(val)
                              for key, val in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def pv(stream, r_t, return_flows=False):
    """
    Calculates the present value of a stream of flows

    Parameters
    ----------
    stream : array-like
        The vector containing the flows.
    r_t : float or array-like
        The interest rate.
    return_flows : bool, optional
        Discounted flows are returned if True. The default is False.

    Returns
    -------
    float
        The present value.

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
    # tt = np.array([-t for t in range(T)])
    # factor_t = (np.ones(T) + r_t)**tt
    factor_t = np.ones(T) / (np.ones(T) + r_t)
    factor_t[0] = 1.
    flows = stream * factor_t.cumprod()
    if return_flows:
        return flows.sum(), flows
    else:
        return flows.sum()


def steps(d, size):
    """
    Returns an array with values that increase in steps.

    A dictionary
        d = {idx_1: val_1, idx_2: val_2,...}
    produces
        np.array([0, 0, val_1,..., val_1, val_2,..., val_2,...])
    where the switches occur at idx_1, idx_2,...

    Requirement 0 <= idx_j < size, for all j is enforced

    Parameters
    ----------
    d : dict
        Mapping from index numbers to values.
    size : int
        Length of the np.array that is returned.

    Returns
    -------
    a : np.array

    """
    a = np.zeros(size)
    if d:  # an empty dictionary is allowed
        locations = sorted(list(d.keys()))
        assert locations[0] >= 0, "An index in dictionary is out of bounds"
        assert locations[-1] < size, "An index in dictionary is out of bounds"
        for i in locations:
            a[i:] = d[i]
    return a


def transition_vector(length, v0, v1, t0, t1, speed=1.0):
    """
    Returns a vector of size <length> that transitions from <v0> to <v1>
    starting at date <t0> and completing the transition at <t1>.
    Speed controls the speed; <speed> = 1 is linear.
    Vector is zero-indexed, so position t corresponds to date t+1.

    Parameters
    ----------
    length : int
        Length of the vector.
    v0 : float
        Value at the start of the transition.
    v1 : float
        Value at the end of the transition.
    t0 : int
        Index number where the transition starts.
    t1 : int
        Index where the transition ends.
    speed : float, optional
        Controls the speed. The default is 1.0

    Returns
    -------
    vector : np.array

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


def save_obj(obj, path, var_names):
    """
    Saves attributes of an object to a csv file

    Parameters
    ----------
    obj : object with attributes
        Any object with attributes that can be put into a pd.Dataframe.
    path : str
        Path where the file will be saved.
    var_names : list
        List of attributes of the object that will be saved.

    Returns
    -------
    None.

    """
    df = pd.DataFrame()
    for var_name in var_names:
        df[var_name] = getattr(obj, var_name)
    df.to_csv(path, index=False)
