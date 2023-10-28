from ast import IsNot
from random import choice
import CoolProp
import CoolProp.CoolProp as CP
import numpy as np
import time
from numpy import meshgrid
from pandas import to_timedelta
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from numpy.linalg import solve
from numpy.linalg import lstsq
from numpy import log
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
import argparse
import textwrap
import pandas as pd
#from sklearn import mixture
import os
import sys
from pathlib import Path
from scipy.io import savemat

from matplotlib import rc
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "font.sans-serif": ["Helvetica"],
    "font.size": 12})

import warnings
warnings.filterwarnings("ignore")

def getUnit(prop):
    return {'h':     '[J/kg]',
            'c':     '[m/s]',
            'v':     '[m^3/kg]',
            'cv':    '[J/kg/K]',
            'cp':    '[J/kg/K]',
            'dPdvT': '[Pa/m^3/kg]',
            's':     '[J/kg/K]',
            'mu':    '[Pa\;s]',
            'k':     '[W/m/K]',
            'p':     '[Pa]',
            'T':     '[K]'}[prop]


def specify_phase(AS, pi, Tj, pcrit, Tcrit):
    if pi > pcrit and Tj > Tcrit:
        AS.specify_phase(CP.iphase_supercritical_gas)
    elif pi > pcrit and Tj < Tcrit:
        AS.specify_phase(CP.iphase_supercritical_liquid)
    elif pi < pcrit and Tj > Tcrit:
        AS.specify_phase(CP.iphase_gas)
    elif pi < pcrit and Tj < Tcrit:
        AS.unspecify_phase()
    return AS

def scalePlot(pr):
    vmin_ = None
    vmax_ = None
    if pr=='mu':
        vmin_=0
        vmax_=0.0001
    elif pr=='k':
        vmin_=0
        vmax_=0.25
    elif pr=='c':
        vmin_=0
        vmax_=500
    elif pr=='cv':
        vmin_=0
        vmax_=1500
    elif pr=='cp':
        vmin_=0
        vmax_=4000
    elif pr=='dPdvT':
        vmin_=-1e12
        vmax_= 0

    return vmin_,vmax_

def parseBool(arg):
    if arg == 'False' or arg == 0 or arg==False:
        ret = False
    elif arg == 'True' or arg == 1 or arg==True:
        ret = True
    return ret

def setREFPROP_PATH(path):
    CP.set_config_string(CP.ALTERNATIVE_REFPROP_PATH, path)


def Eng(x):
    return '\t' + np.format_float_scientific(x, exp_digits=3, precision=7, unique=False).upper()


def vec2str(vec):
    string = ''
    for i, el in enumerate(vec):
        if i % 5 == 0:
            string += ' \n'
        string += Eng(el)
    return string


def TSat2spinodal(AS, FluidName, backend, T_sat):
    # Inspired: https://github.com/CoolProp/CoolProp/blob/master/doc/notebooks/Maxwell_Loop.ipynb
    # Paper: https://link.springer.com/article/10.1134/S0036024406040030

    FluidName
    myIdx = 0
    # Constants
    eps = 1e-6
    kilo = 1e3
    Mega = 1e6
    golden = (1 + 5 ** 0.5) / 2
    width = 12.5
    # Constants, triple and critical data
    R = PropsSI('GAS_CONSTANT',FluidName)
    MM = PropsSI('MOLAR_MASS',FluidName)
    Rs = R/MM
    T_crt = PropsSI('T_CRITICAL',FluidName)
    T_trp = PropsSI('T_TRIPLE',FluidName)
    p_crt = PropsSI('P_CRITICAL',FluidName)
    p_trp = PropsSI('P_TRIPLE',FluidName)
    p_max = PropsSI('P_MAX',FluidName)
    d_crt = PropsSI('RHOMASS_CRITICAL',FluidName)
    v_crt = 1/d_crt
    d_trp_liq = PropsSI('D','T',T_trp,'Q',0,FluidName)
    d_trp_vap = PropsSI('D','T',T_trp,'Q',1,FluidName)

    # Calculation of the coefficients for the metastable region interpolation happens in this cell
    nPoints = T_sat.size
    # empty arrays
    # vap side
    delta_vap = np.empty(nPoints)
    tau_vap = np.empty(nPoints)
    p_vap = np.empty(nPoints)
    d_vap = np.empty(nPoints)
    v_vap = np.empty(nPoints)
    f_vap = np.empty(nPoints)
    dP_dD_T_vap = np.empty(nPoints)
    d2P_dD2_T_vap = np.empty(nPoints)
    d2P_dDdT_vap = np.empty(nPoints)
    # liq side
    delta_liq = np.empty(nPoints)
    tau_liq = np.empty(nPoints)
    p_liq = np.empty(nPoints)
    d_liq = np.empty(nPoints)
    v_liq = np.empty(nPoints)
    f_liq = np.empty(nPoints)
    dP_dD_T_liq = np.empty(nPoints)
    d2P_dD2_T_liq = np.empty(nPoints)
    d2P_dDdT_liq = np.empty(nPoints)
    # metastable coeffs:
    AShape = (8,8)
    A = np.empty(AShape)
    b = np.empty(8)
    xShape = (nPoints,8)
    x = np.empty(xShape)

    #HEOS = CP.AbstractState(backend, FluidNameAS)
    HEOS = AS
    # get values from CoolProp
    for idx in range(0,nPoints):
        # AT the vap line
        HEOS.update(CP.QT_INPUTS, 1, T_sat[idx])
        delta_vap[idx] = HEOS.delta()
        tau_vap[idx] = HEOS.tau()
        p_vap[idx] = HEOS.p()
        d_vap[idx] = HEOS.rhomass()
        f_vap[idx] = Rs*T_sat[idx]*( HEOS.alpha0() + HEOS.alphar() )
        dP_dD_T_vap[idx] = HEOS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
        d2P_dD2_T_vap[idx] = HEOS.second_partial_deriv(CP.iP, CP.iDmass, CP.iT, CP.iDmass, CP.iT)
        d2P_dDdT_vap[idx] = HEOS.second_partial_deriv(CP.iP, CP.iDmass, CP.iT, CP.iT, CP.iDmass)

        # AT the liq line
        HEOS.update(CP.QT_INPUTS, 0, T_sat[idx])
        delta_liq[idx] = HEOS.delta()
        tau_liq[idx] = HEOS.tau()
        p_liq[idx] = HEOS.p()
        d_liq[idx] = HEOS.rhomass()
        f_liq[idx] = Rs*T_sat[idx]*( HEOS.alpha0() + HEOS.alphar() )
        # f_liq[idx] = HEOS.umass() - T_sat[idx]*HEOS.smass()
        dP_dD_T_liq[idx] = HEOS.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
        d2P_dD2_T_liq[idx] = HEOS.second_partial_deriv(CP.iP, CP.iDmass, CP.iT, CP.iDmass, CP.iT)
        d2P_dDdT_liq[idx] = HEOS.second_partial_deriv(CP.iP, CP.iDmass, CP.iT, CP.iT, CP.iDmass)

        # calculate metastable coeffs by solving Ax=b
        A = np.array([  [1/tau_vap[idx], -1/delta_vap[idx]/tau_vap[idx],  log(delta_vap[idx]),          delta_vap[idx],     delta_vap[idx]**2/2,       delta_vap[idx]**3/3,         delta_vap[idx]**4/4,        delta_vap[idx]**5/5 ],
                        [1/tau_liq[idx], -1/delta_liq[idx]/tau_liq[idx],  log(delta_liq[idx]),          delta_liq[idx],     delta_liq[idx]**2/2,       delta_liq[idx]**3/3,         delta_liq[idx]**4/4,        delta_liq[idx]**5/5 ],
                        [             0,             d_crt/tau_vap[idx], d_crt*delta_vap[idx], d_crt*delta_vap[idx]**2, d_crt*delta_vap[idx]**3,    d_crt*delta_vap[idx]**4,    d_crt*delta_vap[idx]**5,    d_crt*delta_vap[idx]**6 ],
                        [             0,             d_crt/tau_liq[idx], d_crt*delta_liq[idx], d_crt*delta_liq[idx]**2, d_crt*delta_liq[idx]**3,    d_crt*delta_liq[idx]**4,    d_crt*delta_liq[idx]**5,    d_crt*delta_liq[idx]**6 ],
                        [             0,                              0,                    1,        2*delta_vap[idx],     3*delta_vap[idx]**2,        4*delta_vap[idx]**3,        5*delta_vap[idx]**4,        6*delta_vap[idx]**5 ],
                        [             0,                              0,                    1,        2*delta_liq[idx],     3*delta_liq[idx]**2,        4*delta_liq[idx]**3,        5*delta_liq[idx]**4,        6*delta_liq[idx]**5 ],
                        [             0,                              0,                    0,                 2/d_crt,  6*delta_vap[idx]/d_crt, 12*delta_vap[idx]**2/d_crt, 20*delta_vap[idx]**3/d_crt, 30*delta_vap[idx]**4/d_crt ],
                        [             0,                              0,                    0,                 2/d_crt,  6*delta_liq[idx]/d_crt, 12*delta_liq[idx]**2/d_crt, 20*delta_liq[idx]**3/d_crt, 30*delta_liq[idx]**4/d_crt ]])
        A = Rs*T_crt*A
        b = np.array([f_vap[idx], f_liq[idx], p_vap[idx], p_liq[idx], dP_dD_T_vap[idx], dP_dD_T_liq[idx], d2P_dD2_T_vap[idx], d2P_dD2_T_liq[idx]])
        x[idx] = solve(A,b)

    pms = np.ones(nPoints)

    N=nPoints-1
    myIdxs = np.linspace(0,N-1,N)

    Tspin = np.zeros_like(myIdxs)
    pspin_liq = np.zeros_like(myIdxs)
    pspin_vap = np.zeros_like(myIdxs)
    rhorspin_liq = np.zeros_like(myIdxs)
    rhorspin_vap = np.zeros_like(myIdxs)

    for i,myIdx in enumerate(myIdxs):
        myIdx = int(myIdx)
        T_iso = T_sat[myIdx]
        tau_iso = T_crt/T_iso

        c = x[myIdx,:]
        d_min = 0.8*d_vap[myIdx]
        # verificar se válido !
        #d_max = PropsSI('D','T',T_iso,'P',p_max,FluidName)
        d_max = d_trp_liq
        rhos = np.linspace(d_min, d_max, num=nPoints)
        deltas = rhos/d_crt

        for idx in range(0,nPoints):
            # stable
            pms[idx]  = Rs*T_crt*d_crt*( 0       +c[1]/tau_iso             +c[2]*deltas[idx]      +c[3]*deltas[idx]**2 +c[4]*deltas[idx]**3   +c[5]*deltas[idx]**4   +c[6]*deltas[idx]**5   +c[7]*deltas[idx]**6   )


        dpdrho_T = np.gradient(pms)/np.gradient(rhos)
        inflection = np.where( ((np.diff(np.sign(dpdrho_T)) != 0)*1) == 1 )[0]
        Tspin[i] = T_iso
        pspin_liq[i] = pms[inflection[0]]
        pspin_vap[i] = pms[inflection[1]]
        rhorspin_liq[i] = (rhos/d_crt)[inflection[0]]
        rhorspin_vap[i] = (rhos/d_crt)[inflection[1]]

    return Tspin, pspin_liq, pspin_vap


class AbstractState(CP.AbstractState):
    def __init__(self, *args, **kwargs):
        self.initPrint()
        self.fluidname = kwargs['fluid']
        self.backend = kwargs['backend']
        super().__init__()

    def initPrint(self):
        pass

    def updateTP(self, T, p):
        self.update(CP.PT_INPUTS, p, T)

    def updateQT(self, Q, T):
        self.update(CP.QT_INPUTS, Q, T)

    def updatePQ(self, p, Q):
        self.update(CP.PQ_INPUTS, p, Q)

    def p_triple(self):
        abs = CP.AbstractState('HEOS', self.fluidname)
        abs.set_mass_fractions(self.get_mass_fractions())

        return abs.trivial_keyed_output(CP.iP_triple)


    def getProp(self, prop):
        try:
            if prop == 'T':
                return self.T()
            if prop == 'p':
                return self.p()
            if prop == 'h':
                return self.hmass()
            if prop == 'c':
                return self.speed_sound()
            if prop == 'v':
                return 1/self.rhomass()
            if prop == 'cv':
                return self.cvmass()
            if prop == 'cp':
                return self.cpmass()
            if prop == 'dPdvT':
                return -self.rhomass()**2*self.first_partial_deriv(CP.iP, CP.iDmass, CP.iT)
            if prop == 's':
                return self.smass()
            if prop == 'k':
                return self.conductivity()
            if prop == 'mu':
                return self.viscosity()
        except:
            return 0


    def coolpropname(self, decimalplaces=False):
        fluidname = self.fluid_names()

        if len(fluidname) > 1:
            molef = self.get_mole_fractions()
            fluid = ''
            for f,m in zip(fluidname,molef):
                if decimalplaces==1:
                    fluid = fluid + f + '[' + '{:.1f}'.format(m) + ']&'
                elif decimalplaces==2:
                    fluid = fluid + f + '[' + '{:.2f}'.format(m) + ']&'
                elif decimalplaces==3:
                    fluid = fluid + f + '[' + '{:.3f}'.format(m) + ']&'
                elif decimalplaces==4:
                    fluid = fluid + f + '[' + '{:.4f}'.format(m) + ']&'
                elif decimalplaces==5:
                    fluid = fluid + f + '[' + '{:.5f}'.format(m) + ']&'
                elif decimalplaces==6:
                    fluid = fluid + f + '[' + '{:.6f}'.format(m) + ']&'
                elif decimalplaces==7:
                    fluid = fluid + f + '[' + '{:.7f}'.format(m) + ']&'
                elif decimalplaces==8:
                    fluid = fluid + f + '[' + '{:.8f}'.format(m) + ']&'
                else:
                    fluid = fluid + f + '[' + str(m) + ']&'

            fluid = fluid[:-1]
        else:
            fluid = fluidname[0]
        return fluid


class RGP:
    # Classe de abstração para uma tabela de propriedades de gas real (RGP)
    def __init__(self):

        pass

    def setInterpKind(self, kind):
        self.interpKind = kind

    def setPath(self, op):
        self.op = op
        os.makedirs(self.op, exist_ok=True)

    def openFile(self):
        self.o = open(self.fname.split('.rgp')[0] +'.out', "w")

    def print(self, *args,**kwargs):
        try:
            end = kwargs['end']
        except:
            end = '\n'
        self.o.write(args[0] + end)
        print(*args,**kwargs)

    def genRGP(self, args):
        # Parsing dos argumentos

        fluid = args['fluid']
        if len(args['massfractions']) > 0:
            massf = [float(mi) for mi in args['massfractions'].split(",")]
        else:
            massf= []
        backend = args['backend']
        if backend == 'REFPROP':
            setREFPROP_PATH(path=args['refprop_path'])
        p = [float(pi) for pi in args['pressures'].split(",")]
        T = [float(pi) for pi in args['temperatures'].split(",")]
        if args['sat_table_range'] is not None:
            Tsat = [float(pi) for pi in args['Sat_table_range'].split(",")]
        else:
            Tsat = None
        NT = int(args['n_temperatures'])
        Np = int(args['n_pressures'])
        Ns = int(args['n_saturation'])
        model = int(args['model'])
        fname = args['output_file']
        meta = parseBool(args['metastable'])
        sat = parseBool(args['sat_table'])
        clip = parseBool(args['clipping'])
        spin = parseBool(args['spinodal'])
        sat_phase =args['sat_phase']
        op =args['output_path']
        kind = args['interpolationKind']

        # Gera o RGP com base nos argumentos
        AS = AbstractState(backend=backend, fluid=fluid)
        if len(massf) > 0:
            AS.set_mass_fractions(massf)

        self.setPath(op)
        self.setFname(fname)
        a = 'python'
        for ai in sys.argv:
            a += ' ' + ai
        self.print(a)

        self.setFluid(AS)
        self.setModel(model)
        self.setUnits(1)
        self.setInterpKind(kind)

        self.setT(Tmin=T[0], Tmax=T[1])

        if Tsat is not None:
            self.setTsat(Tminsat=Tsat[0], Tmaxsat=Tsat[1])
        self.setp(pmin=p[0], pmax=p[1])
        self.setDomain(NT=NT, Np=Np)
        self.setNs(Ns=Ns)

        self.setMetastable(meta)
        self.setSpinodal(spin)
        self.setSatTable(sat)
        self.setClipping(clip)
        self.setSatPhase(sat_phase)

        self.genSuperTables()
        self.genSatTable()
        self.genMetaTables()

        self.writeHeader()
        self.writeData()
        self.writeSuperTables()
        self.writeSuperTablesMatrices()
        self.writeSatTable()
        self.writeSatTableMatrices()
        self.closeFile()
        self.etime()

        pformat = args['plotFormat']

        if args['plotTS'] is not None:
            self.plotTs(args['plotTS'], isobars=10, save=True, format=pformat)
        if args['plotPH'] is not None:
            self.plotPH(args['plotPH'], isothermals=10, save=True, format=pformat)
        if args['plotPT'] is not None:
            self.plotPT(args['plotPT'], save=True, format=pformat)

        show = parseBool(args['plotShow'])
        if show:
            plt.show()

        self.print('END')
        return self

    def setFluid(self, AS):
        self.fluid = AS
        #self.name = '&'.join(self.fluid.fluid_names())
        self.name = self.fluid.fluidname
        self.coolpropname = self.fluid.coolpropname()
        be = self.fluid.backend_name()
        if be == 'REFPROPBackend':
            be += ' ' + CP.get_global_param_string("REFPROP_version")
        self.description = self.coolpropname + \
            ' [' + be + ', CoolProp ' + CoolProp.__version__ + ']'
        self.index = self.name

        self.pcrit = self.fluid.p_critical()
        self.Tcrit = self.fluid.T_critical()

        self.Ttrip = self.fluid.Ttriple()
        self.ptrip = self.fluid.p_triple()

        self.R = self.fluid.gas_constant()/self.fluid.molar_mass()
        self.table = {}
        self.properties = ['h', 'c', 'v', 'cv', 'cp', 'dPdvT', 's', 'mu', 'k']

        self.print('RGP table for fluid: ')
        self.print(self.description)

        self.stime = time.time()

        self.interpMeta = 'p'

        self.mixture = False
        if len(self.fluid.fluid_names()) > 1:
            self.mixture = True


    def closeFile(self):
        self.close(self.f)


    def setFname(self, fn):
        self.fname = self.op+fn
        self.f = open(self.fname, 'w')
        self.openFile()

    def setClipping(self, val):
        self.clipping = val

    def setSatTable(self, val):
        self.sat_table = val

    def setSatPhase(self, val):
        self.sat_phase = val

    def setMetastable(self, val):
        self.metastable = val

    def setSpinodal(self, val):
        self.spinodal = val

    def setT(self, Tmin, Tmax):
        self.Tmin = Tmin
        self.Tmax = Tmax

        self.Tminsat = max(self.Ttrip, Tmin)
        self.Tmaxsat = min(self.Tcrit, Tmax)

        self.print(f'T: [{self.Tmin}: {self.Tmax}] K')

    def setTsat(self, Tminsat, Tmaxsat):
        self.Tminsat = Tminsat
        self.Tmaxsat = Tmaxsat

    def setp(self, pmin, pmax):
        self.pmin = pmin
        self.pmax = pmax

        self.print(f'p: [{self.pmin/1e3}: {self.pmax/1e3}] kPa')

    def alocSuperTables(self):
        # Alocando tabelas
        self.print('Allocating Super tables... ', end='\t')
        stime = time.time()

        for prop in self.properties:
            self.table[prop] = np.zeros((self.Np, self.NT))
            self.table[prop+'sat'] = np.zeros(self.Np)

        self.table['Tsat'] = np.zeros(self.Np)
        self.table['psat'] = np.zeros(self.Np)

        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')

    def alocSatTable(self):
        self.print('Allocating Sat Table... ', end='\t')
        stime = time.time()

        for prop in ['T', 'p'] + self.properties:
            self.table[prop+'satliq'] = np.zeros(self.Ns)
            self.table[prop+'satvap'] = np.zeros(self.Ns)

        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')

    def alocMetaTable(self):
        self.print('Allocating Metastable Table... ', end='\t')
        stime = time.time()

        for prop in ['T', 'p'] + self.properties:
            self.table[prop+'spinliq'] = np.zeros(self.Ns)
            self.table[prop+'spinvap'] = np.zeros(self.Ns)
            self.table[prop+'spin'] = np.zeros(self.Np)
            if prop not in ['T','p']:
                self.table[prop+'meta'] = self.table[prop]

        #self.table['Tspin'] = np.zeros(self.Np)
        #self.table['pspin'] = np.zeros(self.Np)

        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')

    def genSuperTables(self):
        self.alocSuperTables()

        self.print('Generating Super Tables... ', end='\n')
        stime = time.time()

        if self.mixture:
            eps = 10
            Ttrip = self.Ttrip
            Tcrit = self.Tcrit
            pcrit = self.pcrit
            T_sat = np.linspace(Ttrip+eps, Tcrit-eps, self.Ns)
            p_liq = np.empty(T_sat.size)
            p_vap = np.empty(T_sat.size)

            for idx, T in enumerate(T_sat):
                self.fluid.update(CP.QT_INPUTS, 1, T_sat[idx])
                p_vap[idx] = self.fluid.p()

                self.fluid.update(CP.QT_INPUTS, 0, T_sat[idx])
                p_liq[idx] = self.fluid.p()


            self.p_spl_liq = UnivariateSpline(np.hstack((T_sat, Tcrit)),
                                        np.hstack((p_liq, pcrit)),k=3,s=0)
            self.p_spl_vap = UnivariateSpline(np.hstack((T_sat, Tcrit)),
                                        np.hstack((p_vap, pcrit)),k=3,s=0)

            self.T_spl_liq = UnivariateSpline(np.hstack((p_liq, pcrit)),
                                np.hstack((T_sat, Tcrit)),k=3,s=0)
            self.T_spl_vap = UnivariateSpline(np.hstack((p_vap, pcrit)),
                                np.hstack((T_sat, Tcrit)),k=3,s=0)

            self.p_sat_liq = np.linspace(self.p_spl_liq(self.Tmin),
                                         self.p_spl_liq(self.Tcrit), self.Ns)
            self.p_sat_vap = np.linspace(self.p_spl_vap(self.Tmin),
                                         self.p_spl_vap(self.Tcrit), self.Ns)

        if self.sat_phase == 'liquid':
            Q = 0
            self.pmin = self.p_spl_liq(self.Tmin)
            self.table['p'] = np.linspace(self.pmin, self.pmax, self.Np)
        elif self.sat_phase == 'gas':
            Q = 1

        for j, p in enumerate(self.table['p']):
            for i, T in enumerate(self.table['T']):
                # Controla se corrige o erro de pressão muito proximo a saturação
                fix = False
                if self.fluid.backend!='REFPROP':
                    fix = True

                # if clipping
                if p < self.pcrit and T < self.Tcrit and self.clipping:
                    self.fluid.updatePQ(p,Q)
                    Tsat = self.fluid.getProp('T')
                    if T > Tsat:
                        self.fluid = specify_phase(self.fluid , p, T, self.pcrit, self.Tcrit)
                        self.fluid.updateTP(T, p)
                # if fix
                elif fix:
                    try:
                        self.fluid = specify_phase(self.fluid , p, T, self.pcrit, self.Tcrit)
                        self.fluid.updateTP(T, p)
                    except:
                        self.fluid.updateQT(1, T)
                        psat = self.fluid.getProp('p')
                        if abs(p - psat)/psat <= 1e-6:
                            if p > psat:
                                self.fluid.updateQT(0, T)
                # except
                else:
                    self.fluid = specify_phase(self.fluid , p, T, self.pcrit, self.Tcrit)
                    try:
                        self.fluid.updateTP(T, p)
                    except:
                        pass


                for prop in self.properties:
                    self.table[prop][j, i] = self.fluid.getProp(prop)
                    #acho que está invertido...
                    #self.table[prop][i, j] = self.fluid.getProp(prop)

            #if p < self.pcrit:  # and T < self.pcrit:
            #    self.fluid.updatePQ(p, 1)
            #else:
            #    self.fluid.updateTP(self.Tcrit, p)

            if self.mixture:
                if p < self.pcrit:
                    if Q == 0:
                        # precisa de correção ajustar intervalo da tabela para conter a curva de saturação do vapor
                        Tsat = self.T_spl_liq(p)
                    elif Q == 1:
                        Tsat = self.T_spl_vap(p)
                    self.fluid = specify_phase(self.fluid , p, Tsat, self.pcrit, self.Tcrit)
                    self.fluid.updateTP(Tsat, p)
                else:
                    self.fluid.updateTP(self.Tcrit, p)
                for prop in ['T', 'p'] + self.properties:
                    self.table[prop+'sat'][j] = self.fluid.getProp(prop)

            else:
                try:
                    if p < self.pcrit:
                        self.fluid.updatePQ(p, Q)
                except:
                    self.fluid = specify_phase(self.fluid , p, T, self.pcrit, self.Tcrit)
                    self.fluid.updateTP(self.Tcrit, p)
                for prop in ['T', 'p'] + self.properties:
                    self.table[prop+'sat'][j] = self.fluid.getProp(prop)

            self.fluid.unspecify_phase()

        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')

    def genMetaTables(self):
        if self.metastable or self.spinodal or self.mixture:
            self.alocMetaTable()

            self.print('Generating Metastable Table... ', end='\t')
            stime = time.time()

            if self.mixture:
                pspin_liq = self.table['psatliq']
                Tspin = self.table['Tsatliq']
            else:
                # curva Spinodal valores discretos
                Tsat = self.table['Tsatliq']

                fluid = self.fluid.coolpropname()

                Tspin, pspin_liq, pspin_vap = TSat2spinodal(self.fluid, fluid, self.fluid.backend, Tsat[1:-1])

                # adiciona o ponto critico aos vetores
                Tspin = np.append(Tspin, self.Tcrit)
                pspin_liq = np.append(pspin_liq, self.pcrit)
                pspin_vap = np.append(pspin_vap, self.pcrit)

                # remove valores de temperatura abaixo do intervalo da tabela
                idxs = np.where(Tspin > self.Tmin)
                Tspin = Tspin[idxs]
                pspin_liq = pspin_liq[idxs]
                pspin_vap = pspin_vap[idxs]

            # cria função para a tempertura na spinodal

            T_spin = interpolate.UnivariateSpline(pspin_liq, Tspin)
            p_spin = interpolate.UnivariateSpline(Tspin, pspin_liq)

            f_spin_p = {}
            f_spin_T = {}
            f_sat_T = {}
            for prop in self.properties:
                y = np.zeros_like(pspin_liq[:])
                for i, (p,T) in enumerate(zip(pspin_liq[:], Tspin[:] )):
                    try:
                        self.fluid = specify_phase(self.fluid , p, T, self.pcrit, self.Tcrit)
                        self.fluid.updateTP(T,p)
                        y[i] =  self.fluid.getProp(prop)
                        self.fluid.unspecify_phase()
                    except:
                        y[i] = np.nan

                idx = np.where(np.isnan(y))
                pspin_liq = np.delete(pspin_liq,idx)
                y = np.delete(y,idx)
                Tspin = np.delete(Tspin,idx)

                f_spin_p[prop] = interpolate.UnivariateSpline(pspin_liq, y)# , bounds_error=False, fill_value=(y[0],y[-1]), kind='cubic')
                f_spin_T[prop] = interpolate.UnivariateSpline(Tspin, y)# , bounds_error=False, fill_value=(y[0],y[-1]), kind='cubic')


            # Gera spinodal no intervalo da tabela
            for j, p in enumerate(self.table['p']):
                if p < self.pcrit:
                    Ts = T_spin(p)
                else:
                    Ts = self.Tcrit

                self.table['Tspin'][j] = Ts
                self.table['pspin'][j] = p
                #self.fluid.updateTP(Ts,p)
                for prop in self.properties:
                    if p >= self.pcrit:
                        self.fluid = specify_phase(self.fluid , p, T, self.pcrit, self.Tcrit)
                        self.fluid.updateTP(Ts, p)
                        self.table[prop+'spin'][j] = self.fluid.getProp(prop)
                    else:
                        self.table[prop+'spin'][j] = f_spin_p[prop](p)
                    self.fluid.unspecify_phase()


            # interpola na região de metaestabilidade
            if self.interpMeta=='p'and self.metastable:
                for j, p in enumerate(self.table['p']):
                    if p <= self.pcrit:
                        for i, T in enumerate(self.table['T']):
                            if T <= self.Tcrit:
                                for prop in self.properties:
                                    pspin = p_spin(T)
                                    self.fluid.updateQT(1,T)
                                    psat = self.fluid.getProp('p')
                                    prop_sat = self.fluid.getProp(prop)
                                    prop_spin = f_spin_p[prop](pspin)
                                    if p < pspin*1.01 \
                                        and p > psat*0.99:

                                        jlow = np.where(self.table['p'] < psat)
                                        plow = self.table['p'][jlow]
                                        jup = np.where(self.table['p'] > pspin)
                                        pup = self.table['p'][jup]
                                        proplow = self.table[prop][jlow,i][0]
                                        propup = self.table[prop][jup,i][0]

                                        Np = 5
                                        if len(plow) < Np:
                                            dp = abs(self.table['p'][1]-self.table['p'][0])
                                            plow = np.linspace(psat*.99-dp*Np,psat*.99, 10)
                                            proplow = np.zeros_like(plow)

                                            for k, pl in enumerate(plow):
                                                self.fluid = specify_phase(self.fluid , pl, T, self.pcrit, self.Tcrit)
                                                self.fluid.updateTP(T, pl)
                                                proplow[k] = self.fluid.getProp(prop)

                                        if self.mixture:
                                            idx=1
                                            x=np.hstack([plow[:-idx],
                                                        pup[idx:]])
                                            y=np.hstack([proplow[:-idx],
                                                         propup[idx:]])
                                        else:
                                            x=np.hstack([plow, pup])
                                            y=np.hstack([proplow, propup])


                                        f = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0],y[-1]), kind=self.interpKind)

                                        # dpdvT ?
                                        self.table[prop+'meta'][j, i] = f(p)

                            self.fluid.unspecify_phase()

            elif self.interpMeta=='T' and self.metastable:
                for i, T in enumerate(self.table['T']):
                    if T <= self.Tcrit:
                        for j, p in enumerate(self.table['p']):
                            if p <= self.pcrit:
                                for prop in self.properties:
                                    Tspin = T_spin(p)
                                    self.fluid.updatePQ(p,1)
                                    Tsat = self.fluid.getProp('T')
                                    if T > Tspin*1.01 \
                                        and T < Tsat*1.01:

                                        prop_sat = self.fluid.getProp(prop)
                                        prop_spin = f_spin_T[prop](T)

                                        x=[Tspin, Tsat]
                                        y=[prop_spin, prop_sat]
                                        f = interpolate.interp1d(x, y, bounds_error=False, fill_value=(y[0],y[-1]))

                                        self.table[prop+'meta'][j, i] = f(T)

            for j, p in enumerate(self.table['psatliq']):
                Ts = T_spin(p)
                self.table['pspinliq'][j] = p
                self.table['Tspinliq'][j] = Ts
                for prop in self.properties:
                    self.table[prop+'spinliq'][j] = f_spin_p[prop](p)

            etime = time.time() - stime
            self.print('{:.4f}'.format(etime) + ' sec')


    def genSatTable(self):
        #self.fluid.unspecify_phase()
        self.alocSatTable()

        self.print('Generating Sat Table... ', end='\t')
        stime = time.time()


        if self.mixture:

            self.fluid.specify_phase(CP.iphase_liquid)
            self.table['psatliq'] = self.p_sat_liq
            for j, p in enumerate(self.table['psatliq']):
                self.fluid.updateTP(self.T_spl_liq(p), p)
                for prop in ['T']+self.properties:
                    self.table[prop+'satliq'][j] = self.fluid.getProp(prop)

            self.fluid.specify_phase(CP.iphase_gas)
            self.table['psatvap'] = self.p_sat_vap
            for j, p in enumerate(self.table['psatvap']):
                self.fluid.updateTP(self.T_spl_vap(p), p)
                for prop in ['T']+self.properties:
                    self.table[prop+'satvap'][j] = self.fluid.getProp(prop)

        else:

            # Liquido
            self.fluid.specify_phase(CP.iphase_liquid)
            self.fluid.updateQT(0, self.Tminsat)
            pminsatliq = self.fluid.p()
            self.fluid.updateQT(0, self.Tmaxsat)
            pmaxsatliq = self.pcrit
            if abs(self.fluid.p() - self.pcrit) < 1e-6:
                self.Tmaxsat = self.Tcrit
                pmaxsatliq = self.fluid.p()
            self.table['psatliq'] = np.linspace(pminsatliq, pmaxsatliq, self.Ns)
            for j, p in enumerate(self.table['psatliq']):
                self.fluid.updatePQ(p, 0)
                for prop in ['T']+self.properties:
                    self.table[prop+'satliq'][j] = self.fluid.getProp(prop)


            # Vapor
            self.fluid.specify_phase(CP.iphase_gas)
            self.fluid.updateQT(1, self.Tminsat)
            pminsatvap = self.fluid.p()
            self.fluid.updateQT(1, self.Tmaxsat)
            pmaxsatvap = self.pcrit
            if abs(self.fluid.p() - self.pcrit) < 1e-6:
                self.Tmaxsat = self.Tcrit
                pmaxsatvap = self.fluid.p()
            self.table['psatvap'] = np.linspace(pminsatvap, pmaxsatvap, self.Ns)


            for j, p in enumerate(self.table['psatvap']):
                self.fluid.updatePQ(p, 1)
                for prop in ['T']+self.properties:
                    self.table[prop+'satvap'][j] = self.fluid.getProp(prop)

        self.fluid.unspecify_phase()


        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')


    def genTables(self, NT, Np):
        self.genSuperTables()
        self.genSatTable()

    def setDomain(self, NT, Np):
        self.print(f'RGP Size: (NT={NT}, Np={Np})')
        self.NT = NT
        self.Np = Np

        # Domínio linearmente espaçado
        self.table['p'] = np.linspace(self.pmin, self.pmax, self.Np)
        self.table['T'] = np.linspace(self.Tmin, self.Tmax, self.NT)

    def writeSatTable(self):
        f = self.f
        if self.sat_table:
            self.print('Writing Sat Table... ', end='\t\t')
            stime = time.time()

            f.write('$$SAT_TABLE \n')
            f.write('\t'+str(self.Np) + '\t4\t9')

            for i, t in enumerate(['p', 'T']):
                if self.spinodal:
                    f.write(vec2str(self.table[t+'spinliq'].flatten()))
                else:
                    f.write(vec2str(self.table[t+'satliq'].flatten()))

            for i, t in enumerate(['p', 'T']):
                f.write(vec2str(self.table[t+'satvap'].flatten()))

            for i, t in enumerate(self.properties):
                if self.spinodal:
                    f.write(vec2str(self.table[t+'spinliq'].flatten()))
                else:
                    f.write(vec2str(self.table[t+'satliq'].flatten()))
                f.write(vec2str(self.table[t+'satvap'].flatten()))



            etime = time.time() - stime
            self.print('{:.4f}'.format(etime) + ' sec')

    def writeSatTableMatrices(self):

        if self.sat_table:
            dict_mat = {}
            self.print('Writing Sat Table Matrices... ', end='\t\t')
            stime = time.time()


            for i, t in enumerate(['p', 'T']):
                if self.spinodal:
                    dict_mat[t+'spinliq'] = self.table[t+'spinliq']
                else:
                    dict_mat[t+'satliq'] = self.table[t+'satliq']

            for i, t in enumerate(['p', 'T']):
                dict_mat[t+'satvap'] = self.table[t+'satvap']

            for i, t in enumerate(self.properties):
                if self.spinodal:
                    dict_mat[t+'spinliq'] = self.table[t+'spinliq']
                else:
                    dict_mat[t+'satliq'] = self.table[t+'satliq']
                dict_mat[t+'satvap']=self.table[t+'satvap']

            fname = Path(self.fname).resolve().with_suffix('').__str__()
            fname += '_sat.mat'
            savemat(file_name=fname,
                    mdict=dict_mat,
                    appendmat=False,
                    format='5',
                    long_field_names=False,
                    do_compression=False,
                    oned_as='row')

            etime = time.time() - stime
            self.print('{:.4f}'.format(etime) + ' sec')

    def writeSuperTables(self):
        f = self.f
        self.print('Writing Super Tables... ', end='\t')
        stime = time.time()

        for i, t in enumerate(self.properties):
            f.write('$TABLE_'+str(i+1)+' \n')
            f.write('\t'+ str(self.NT) + '\t' + str(self.Np))
            f.write(vec2str(self.table['T']))
            f.write(vec2str(self.table['p']))


            if self.metastable:
                f.write(vec2str(self.table[t+'meta'].flatten()))
            else:
                f.write(vec2str(self.table[t].flatten()))

            f.write(vec2str(self.table['Tsat'].flatten()))
            f.write(vec2str(self.table[t+'sat'].flatten()))

            f.write(' \n')

        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')

    def writeSuperTablesMatrices(self):
        dict_mat = {}
        self.print('Writing Super Tables Matrices... ', end='\t')
        stime = time.time()

        dict_mat={
            'T': self.table['T'],
            'p': self.table['p']}
        for i, t in enumerate(self.properties):

            if self.metastable:
                dict_mat[t] = self.table[t+'meta']
            else:
                dict_mat[t] = self.table[t]

            dict_mat['Tsat'] = self.table['Tsat']
            dict_mat[t+'sat'] = self.table[t+'sat']

        savemat(file_name=Path(self.fname).with_suffix('.mat'),
                mdict=dict_mat,
                appendmat=False,
                format='5',
                long_field_names=False,
                do_compression=False,
                oned_as='row')

        etime = time.time() - stime
        self.print('{:.4f}'.format(etime) + ' sec')

    def close(self, f):
        self.print(f'RGP file written... \t\t{f.name}')
        f.close()

    def etime(self):
        self.etime = time.time() - self.stime
        self.print('Total elapsed time... \t\t', end='')
        self.print('{:.4f}'.format(self.etime) + ' sec')
        return self.etime

    def writeParams(self):
        f = self.f
        be = self.fluid.backend_name()
        if be == 'REFPROPBackend':
            be += ' ' + CP.get_global_param_string("REFPROP_version")
        coolpropname = self.fluid.coolpropname(decimalplaces=2)
        desc = coolpropname + \
            ' [' + be + ', CoolProp ' + CoolProp.__version__ + ']'

        f.write(f'$$${self.name[:8]} \n')
        f.write('\t1 \n')
        f.write('$$PARAM \n')
        f.write('\t26 \n')
        f.write('DESCRIPTION \n')
        f.write(f'{desc[:50]} \n')
        f.write('NAME \n')
        f.write(f'{self.name[:8]} \n')
        f.write('INDEX \n')
        f.write(f'{self.name[:8]} \n')
        f.write('MODEL \n')
        f.write(f'\t{self.model} \n')
        f.write('UNITS \n')
        f.write(f'\t{self.units} \n')
        f.write('PMIN_SUPERHEAT \n')
        f.write(f'{Eng(self.pmin)} \n')
        f.write('PMAX_SUPERHEAT \n')
        f.write(f'{Eng(self.pmax)} \n')
        f.write('TMIN_SUPERHEAT \n')
        f.write(f'{Eng(self.Tmin)} \n')
        f.write('TMAX_SUPERHEAT \n')
        f.write(f'{Eng(self.Tmax)} \n')
        f.write('TMIN_SATURATION \n')
        f.write(f'{Eng(self.Tminsat)} \n')
        f.write('TMAX_SATURATION \n')
        f.write(f'{Eng(self.Tmaxsat)} \n')
        f.write('SUPERCOOLING \n')
        f.write(f'{Eng(0)} \n')
        f.write('P_CRTICAL \n')
        f.write(f'{Eng(self.pcrit)} \n')
        f.write('P_TRIPLE \n')
        f.write(f'{Eng(self.ptrip)} \n')
        f.write('T_CRTICAL \n')
        f.write(f'{Eng(self.Tcrit)} \n')
        f.write('T_TRIPLE \n')
        f.write(f'{Eng(self.Ttrip)}\n')
        f.write('GAS_CONSTANT \n')
        f.write(f'{Eng(self.R)} \n')
        for i in range(1, 10):
            f.write(f'TABLE_{i} \n')
            f.write(f'\t{self.NT}\t{self.Np} \n')
        if self.sat_table:
            f.write(f'SAT_TABLE \n')
            f.write(f'\t{self.Ns}\t4\t9 \n')

    def writeHeader(self):
        f = self.f
        f.write('$$$$HEADER \n')
        self.writeParams()

    def writeData(self):
        f = self.f
        f.write('$$$$DATA \n')
        self.writeParams()
        f.write(f'$$SUPER_TABLE \n')
        f.write('\t9 \n')

    def setModel(self, m):
        # 1,2,3 are available
        # 1: multiphase non-equilibrium
        # 2: multiphase equilibrium
        # 3: singlephase
        self.model = m

    def setUnits(self, u):
        # system of units 1,2,3,4 or 5
        # 1. UNITS=1 (kg, m, s, K)
        # 2. UNITS=2 (g, cm, s, K)
        # 3. UNITS=3 (lbm, in, s, R)
        # 4. UNITS=4 (slugs, ft, s, R)
        # 5. UNITS=5 (slugs, in, s, R)
        self.units = u

    def setNs(self, Ns):
        # Define numero de pontos nas tabelas de saturação
        self.Ns = Ns

    def setSatCurve(self):
        if self.spinodal:
            self.table['Tsat'] = self.table['Tspin']
            self.table['psat'] = self.table['pspin']


    def plotPT(self, prop, cbarlabel='', ax=None, sct=False, save=False, format='png'):
        if prop == 'all':
            for pr in self.properties:
                self.plotPT(pr, cbarlabel, ax, sct, save, format)
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.gca()
            meta = ''
            if self.metastable:
                meta = 'meta'
            p, T = np.meshgrid(self.table['p'], self.table['T'])

            phi = self.table[prop+meta].T
            #vmi, vma = scalePlot(prop)
            cm = ax.pcolormesh(T, p, phi)#, vmin=vmi, vmax=vma)
            cb = fig.colorbar(cm, ax=ax)
            cb.set_label(r"${}\;{}$".format(prop, getUnit(prop)))
            if sct:
                ax.scatter(T, p, color='k',s=0.5)
            sat = 'sat'
            if self.spinodal:
                sat = 'spin'
            #ax.plot(self.table['T'+sat+'liq'], self.table['p'+sat+'liq'], color='r')
            #ax.plot(self.table['Tspin'+sat], self.table['p'+sat], color='orange', label='spinodal')
            if self.metastable and not self.mixture:
                ax.plot(self.table['Tspinliq'], self.table['pspinliq'], color='orange', ls=':', label='spinodal liquid')

            if self.mixture:
                ax.plot(self.table['Tsatliq'], self.table['psatliq'], color='orange', label='saturation liquid')
                ax.plot(self.table['Tsatvap'], self.table['psatvap'], color='orange', label='saturation vapor')
            else:
                ax.plot(self.table['Tsatliq'], self.table['psatliq'], color='orange', label='saturation')
                ax.plot(self.table['Tsatvap'], self.table['psatvap'], color='orange')

            ax.set_xlabel(r"${}\;{}$".format('T', getUnit('T')))
            ax.set_ylabel(r"${}\;{}$".format('p', getUnit('p')))
            ax.set_ylim([self.pmin,self.pmax])
            ax.set_xlim([self.Tmin,self.Tmax])
            ax.legend()

            if save:
                plt.savefig(self.fname.split('.rgp')[0]+'_PT_'+prop + '.' + format)
            return ax

    def plotPH(self, prop, cbarlabel='', ax=None,  sct=False, save=False, isothermals=False, format='png'):
        if prop == 'all':
            props = [p for p in self.properties if p != 'h']
            for pr in props:
                self.plotPH(pr, cbarlabel, ax, sct, save, isothermals, format)
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.gca()
            meta = ''
            if self.metastable:
                meta = 'meta'
            p, T = np.meshgrid(self.table['p'], self.table['T'])
            h = self.table['h'+meta].T
            if prop == 'T':
                phi = T
            else:
                phi = self.table[prop+meta].T
            cm = ax.pcolormesh(h, p, phi)
            cb = fig.colorbar(cm, ax=ax)
            cb.set_label(r"${}\;{}$".format(prop, getUnit(prop)))
            if sct:
                ax.scatter(h, p, color='k',s=0.5)

            # isotermicas
            if isothermals:
                n = int(self.table['T'].size/isothermals)
                for i in range(0,h[:,0].size):
                    if i%n==0:
                        ax.plot(h[i,:],p[i,:], ls=':', color='r')

            if self.spinodal and not self.mixture:
                ax.plot(self.table['hspinliq'], self.table['pspinliq'], color='orange', ls=':', label='spinodal liquid')

            if self.mixture:
                ax.plot(self.table['hsatliq'], self.table['psatliq'], color='orange', label='saturation liquid')
                ax.plot(self.table['hsatvap'], self.table['psatvap'], color='orange', label='saturation vapor')
            else:
                ax.plot(self.table['hsatliq'], self.table['psatliq'], color='orange', label='saturation')
                ax.plot(self.table['hsatvap'], self.table['psatvap'], color='orange')


            ax.set_xlabel(r"${}\;{}$".format('h', getUnit('h')))
            ax.set_ylabel(r"${}\;{}$".format('p', getUnit('p')))
            ax.set_ylim([self.pmin,self.pmax])
            ax.set_xlim([np.min(h),np.max(h)])
            ax.legend()
            if save:
                plt.savefig(self.fname.split('.rgp')[0]+'_ph_'+prop+'.' + format)
            return ax

    def plotTs(self, prop, cbarlabel='', ax=None, sct=False, save=False, isobars=False, format='png'):
        if prop == 'all':
            props = [p for p in self.properties if p != 's']
            for pr in props:
                self.plotTs(pr, cbarlabel, ax, sct, save, isobars, format)
        else:
            if ax is None:
                fig = plt.figure()
                ax = fig.gca()
            meta = ''
            if self.metastable:
                meta = 'meta'
            p, T = np.meshgrid(self.table['p'], self.table['T'])

            s = self.table['s'+meta].T
            if prop == 'p':
                phi = p
            else:
                phi = self.table[prop+meta].T

            cm = ax.pcolormesh(s, T, phi)
            cb = fig.colorbar(cm, ax=ax)
            cb.set_label(r"${}\;{}$".format(prop, getUnit(prop)))
            if sct:
                ax.scatter(s, T, color='k',s=0.5)



            if self.spinodal and not self.mixture:
                ax.plot(self.table['sspinliq'], self.table['Tspinliq'], color='orange', ls=':', label='spinodal liquid')

            if self.mixture:
                ax.plot(self.table['ssatliq'], self.table['Tsatliq'], color='orange', label='saturation liquid')
                ax.plot(self.table['ssatvap'], self.table['Tsatvap'], color='orange', label='saturation vapor')
            else:
                ax.plot(self.table['ssatliq'], self.table['Tsatliq'], color='orange', label='saturation')
                ax.plot(self.table['ssatvap'], self.table['Tsatvap'], color='orange')

             # isobaricas
            if isobars:
                n = int(self.table['p'].size/isobars)
                for i in range(0,s[0,:].size):
                    if i%n==0:
                        ax.plot(s[:,i],T[:,i], ls=':', color='r')

            ax.set_xlabel(r"${}\;{}$".format('s', getUnit('s')))
            ax.set_ylabel(r"${}\;{}$".format('T', getUnit('T')))
            ax.set_ylim([self.Tmin,self.Tmax])
            ax.set_xlim([np.min(s),np.max(s)])
            ax.legend()

            if save:
                plt.savefig(self.fname.split('.rgp')[0]+'_Ts_'+prop+'.' + format)
            return ax

desc=textwrap.dedent('''\
         RGP Table Generator v.0
         -----------------------
             - Support for CoolProp and REFPROP backends
             - Able to create a liquid-like metastable region
             - Capable of replacing saturation curve with spinodal line
             - Sucessfully tested for CO2

         Sample command to generate a RGP table for CO2
         ----------------------------------------------

            python3 RGP.py -f 'CO2' -b 'REFPROP' -p 3e6,10e6 -T 250,350 -rp /home/ppiper/MEGA/refprop -c False -nT 100 -np 100 -ns 100 -mo 3 -me True -st True -sf gas -sp True -op ./outputs/ -o CO2_100_100.rgp

             - python3 should match the python 3 interpreter alias of your operational system

            Mixtures only working with `REFPROP` backend, in this case fluid name should be given as the AbstractState string standard, and mass fractions as a list `0,1`, example:

            - python3.8 RGP_.py -f 'CO2&Methane' -mf 0.95,0.05 -b REFPROP -p 1e6,20e6 -T 250,500 -rp /home/ppiper/MEGA/refprop -nT 100 -np 100 -me True -op ./outputs/ -o CO2_50_50_COOLPROP.rgp

         Dependencies
         ------------

            -CoolProp==6.4.1
            -matplotlib==3.1.2
            -numpy==1.17.4
            -pandas==1.3.5
            -scipy==1.7.3
         ''')

ap = argparse.ArgumentParser(description=desc,formatter_class=argparse.RawDescriptionHelpFormatter)#,usage='%(prog)s [optional argument] value')
ap.add_argument("-f", "--fluid", required=True, help="CoolProp fluidname")
ap.add_argument("-b", "--backend", required=False, help="CoolProp backend e.g. (HEOS, REFPROP)", default='HEOS', choices=['HEOS','REFPROP', 'SRK', 'PR'])
ap.add_argument("-rp", "--refprop_path", required=False, help="REFPROP install path")
ap.add_argument("-p", "--pressures", required=True, help="Pressure range for the RGP table [Pa] e.g. 'pmin','pmax' ")
ap.add_argument("-T", "--temperatures", required=True, help="Temperature range for the RGP table [K] e.g. 'Tmin','Tmax' ")
ap.add_argument("-np", "--n_pressures", required=True, help="Number of points to discretize pressure range")
ap.add_argument("-nT", "--n_temperatures", required=True, help="Number of points to discretize temperature range")
ap.add_argument("-ns", "--n_saturation", required=False, help="Number of points to discretize saturation tables")
ap.add_argument("-mo", "--model", required=False, help="ANSYS RGP model (3 - single phase, 2 - multiphase equilibrium, 1 - multiphase non-equilibrium)", choices=['1','2','3'], default='3')
ap.add_argument("-o", "--output_file", required=True, help="Output file name")
ap.add_argument("-op", "--output_path", required=False, help="Output path name", default='./')
ap.add_argument("-me", "--metastable", required=False, help="Turn metastable region on and off", default='False', choices=['True','False'])
ap.add_argument("-sp", "--spinodal", required=False, help="Change the saturation line by the spinodal", default='False', choices=['True','False'])
ap.add_argument("-sf", "--sat_phase", required=False, help="Specify phase at saturation curve", default='gas', choices=['gas','liquid'] )
ap.add_argument("-st", "--sat_table", required=False, help="Turn SAT_TABLE on and off", default='False', choices=['True','False'])
ap.add_argument("-Tsat", "--sat_table_range",required=False, help="Temperature range for Saturation Table", default=None)
ap.add_argument("-mf", "--massfractions",required=False, help="Mass fraction of mixtures", default=[])
ap.add_argument("-c", "--clipping",required=False, help="Clip thermodynamic properties at saturation curve", default='False')
ap.add_argument("-pPT", "--plotPT",required=False, help="Plot PT diagram for specified properties", default=None, choices=['h', 'c', 'v', 'cv', 'cp', 'dPdvT', 's', 'mu', 'k', 'all'])
ap.add_argument("-pTS", "--plotTS",required=False, help="Plot Ts diagram for specified properties", default=None, choices=['h', 'c', 'v', 'cv', 'cp', 'dPdvT', 's', 'mu', 'k', 'all'])
ap.add_argument("-pPH", "--plotPH",required=False, help="Plot ph diagram for specified properties", default=None, choices=['h', 'c', 'v', 'cv', 'cp', 'dPdvT', 's', 'mu', 'k', 'all'])
ap.add_argument("-ps", "--plotShow",required=False, help="Enable/disable if plot should be shown", default='False', choices=['True','False'])
ap.add_argument("-pf", "--plotFormat",required=False, help="Imagem file format for ploting", default='png', choices=['eps','jpeg','jpg','pdf','pgf','png','ps','raw','rgba','svg','svgz','tif','tiff'])
ap.add_argument("-ik", "--interpolationKind",required=False, help="Kind of interpolation method", default='cubic', choices=['linear','cubic'])

def mainSample():
    # Sample code to use Thetab as a python module
    N = 100
    fluid = 'Methane&CO2'
    #mf = '0.1,0.9'
    #fluid = 'CO2'
    #mf = []
    backend = 'REFPROP'


    mfs = ['0.1,0.9',
           '0.2,0.8',
           '0.3,0.7',
           '0.4,0.7',
           '0.5,0.5',
           '0.6,0.4',
           '0.7,0.3',
           '0.8,0.2',
           '0.9,0.1']

    for mf in mfs:
        fname = fluid + '_'+str(N)+'x'+str(N)+'_'+ backend + '_' + str(mf.split(',')[0])
        print()
        print('Fracao massica de CH4: ', mf)
        print('----------------------')
        args = vars(ap.parse_args(['-f', fluid ,
                                '-b', backend ,
                                '-p','3e6,10e6',
                                '-T','250,350',
                                '-mf', mf,
                                '-rp','/home/ppiper/MEGA/refprop',
                                '-c','False',
                                '-nT',str(N),
                                '-np',str(N),
                                '-ns',str(N),
                                '-mo','3',
                                '-op','outputs_' + str(mf.split(',')[0]) + '/',
                                '-me','True',
                                '-st','True',
                                '-sf','gas',
                                '-sp','True',
                                '-o', fname + '.rgp',
                                'ik', 'linear']))


        rgp = RGP()
        rgp.genRGP(args)

        print('----------------------')

    rgp.plotTs('all', isobars=10, save=True)
    rgp.plotPH('all', isothermals=10, save=True)
    rgp.plotPT('all', save=True)
    #plt.show()

def main():
    # Default use of Thetab  using the CLI
    args = vars(ap.parse_args())

    args['n_saturation'] = args['n_pressures']
    if(args['metastable'] == 'True'):
        args['spinodal'] == 'True'
    elif(args['metastable'] == 'False'):
        args['spinodal'] == 'False'

    rgp = RGP()
    rgp.genRGP(args)



    #rgp.plotTs('all', isobars=10, save=True)
    #rgp.plotPH('all', isothermals=10, save=True)
    #rgp.plotPT('all', save=True)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        mainSample()
