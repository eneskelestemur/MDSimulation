import numpy as np
from openmm import unit

kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA

def kd_to_delta_g(kd, T=300*unit.kelvin):
    return unit.MOLAR_GAS_CONSTANT_R*T*np.log(kd.value_in_unit(unit.molar))

def delta_g_to_kd(delta_g, T=300*unit.kelvin):
    return np.exp(delta_g/(unit.MOLAR_GAS_CONSTANT_R*T))*unit.molar

def free_energy_to_partition(g, T=300*unit.kelvin):
    return np.exp(-g/(kB*T))

def partition_to_free_energy(partition, T=300*unit.kelvin):
    return -kB*T*np.log(partition)

def k_to_pchembl(k):
    return -np.log10(k.value_in_unit(unit.molar))

def pchembl_to_kd(pchembl):
    return (10**(-pchembl)) * unit.molar

def pchembl_to_delta_g(pchembl, T=300*unit.kelvin):
    return kd_to_delta_g(pchembl_to_kd(pchembl), T)

def delta_g_to_pchembl(delta_g, T=300*unit.kelvin):
    return k_to_pchembl(delta_g_to_kd(delta_g, T))