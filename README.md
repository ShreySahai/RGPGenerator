

RGP Table Generator v.0
-----------------------
- Support for CoolProp and REFPROP backends
- Able to crate a liquid-like metastable region 
- Capable of replacing saturation curve with spinodal line
- Sucessfully tested for CO2

Dependencies
------------
- CoolProp==6.4.2dev 
- matplotlib==3.5.1
- numpy==1.22.1
- pandas==1.4.0
- scipy==1.7.3

Sample command to generate a RGP table for CO2
----------------------------------------------
```python3 RGP.py -f CO2 -b HEOS -p 0.518e6,30e6 -T 230,520 -np 100 -nT 100 -ns 100 -mo 1 -sat True -me True -sp True -o out.rgp```

`python3` should match the python 3 interpreter alias of your operational system

Mixtures only working with `REFPROP` backend, in this case fluid name should be given as the AbstractState string standard, and mass fractions as a list `0,1`, example:

```python3.8 RGP_.py -f 'CO2&Methane' -mf 0.95,0.05 -b REFPROP -p 1e6,20e6 -T 250,500 -rp /home/ppiper/MEGA/refprop -nT 100 -np 100 -me True -op ./outputs/ -o CO2_50_50_COOLPROP.rgp```


Command Line Options
--------------------
```
optional arguments:
  -h, --help            show this help message and exit
  -f , --fluid          CoolProp fluidname
  -b , --backend        CoolProp backend e.g. (HEOS, REFPROP)
  -rp , --refprop_path 
                        REFPROP install path
  -p , --pressures      Pressure range for the RGP table
  -T , --temperatures   Temperature range for the RGP table
  -np , --n_pressures   Number of points to discretize pressure range
  -nT , --n_temperatures 
                        Number of points to discretize temperature range
  -o , --output_file    Output file name
  -me , --metastable    Turn metastable region on and off e.g. (True, False)
```
