from cProfile import label
import CoolProp.CoolProp as cp
from CoolProp.CoolProp import PropsSI 
from CoolProp.CoolProp import AbstractState
from numpy import linspace, zeros, zeros_like, size
import matplotlib.pyplot as plt

fluid = AbstractState ('HEOS', 'CO2')

#Pressure range
Ps= linspace (7.38, 12, 100)*1e6
#Entrophy range
ss= linspace (700, 2100, 10)

#Properties Determination 
h= zeros((size(Ps),size(ss)))
c= zeros((size(Ps),size(ss)))
v= zeros((size(Ps),size(ss)))
CP= zeros((size(Ps),size(ss)))
CV=zeros((size(Ps),size(ss)))
dPdV=zeros((size(Ps),size(ss)))
mu=zeros((size(Ps),size(ss)))
k=zeros((size(Ps),size(ss)))

for j,s in enumerate (ss) :
    for i, P in enumerate (Ps) :
        fluid.update (cp.PSmass_INPUTS, P, s)
        h[i, j]=fluid.hmass () 
        c[i, j]=fluid.speed_sound ()
        #c[i, j]=PropsSI ("A", "P", P, "Smass", s, 'CO2')
        v[i, j]=1/(fluid.rhomass () )
        CP[i, j]=fluid.cpmass ()
        CV[i, j]=fluid.cvmass ()
        dPdV [i, j]=-fluid.rhomass ()**2*fluid.first_partial_deriv(cp.iP,cp.iDmass, cp.iT)
        mu [i, j]=fluid.viscosity ()
        k [i, j]=fluid.conductivity ()

#Plots

#Entalpy
for j,s in enumerate (ss) :
    plt.plot (Ps,h [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('Entalpy [j/kg]')
    plt.title('Entalphy', fontdict=None, loc='center', pad=None)
    plt.legend()
plt.figure ()

#Speed Sound
for j,s in enumerate (ss) :
    plt.plot (Ps,c [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('Speed of sound [m/s]')
plt.legend()
plt.figure ()

#Density
for j,s in enumerate (ss) :
    plt.plot (Ps,v [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('specific volume [m^3/kg]')
plt.legend()
plt.figure ()

#CP
for j,s in enumerate (ss) :
    plt.plot (Ps,CP [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('cp [j/kg.K]')
plt.legend()
plt.figure ()

#CV
for j,s in enumerate (ss) :
    plt.plot (Ps,CV [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('cv [j/kg.K]')
plt.legend()
plt.figure ()

#dP/Dv
for j,s in enumerate (ss) :
    plt.plot (Ps,dPdV [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('dP/dv [j/kg.K]')
plt.legend()
plt.figure ()

#mu
for j,s in enumerate (ss) :
    plt.plot (Ps,mu [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('mu [Pa-s]')
plt.legend()
plt.figure ()

#k
for j,s in enumerate (ss) :
    plt.plot (Ps,k [:,j],label=str(s))
    plt.xlabel('Pressure [Pa]')
    plt.ylabel('k [kW/m/K]')
plt.legend()
#plt.show ()

