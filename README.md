

RGP Table Generator v.0
-----------------------
- Support for CoolProp and REFPROP backends
- Able to crate a liquid-like metastable region 
- Capable of replacing saturation curve with spinodal line
- Sucessfully tested for Air,too

Dependencies
------------
- CoolProp==6.4.2dev 
- matplotlib==3.5.1
- numpy==1.22.1
- pandas==1.4.0
- scipy==1.7.3



Installation
------------

Clone this git repository

```console
git clone https://github.com/ShreySahai/RGPGenerator.git
```
Move to root path of git repository

Run RGP.py

```console
python RGP.py -f Air -p 1e5,10e6 -T 150,1000 -np 20 -nT 20 -ns 20 -Tsat 80,82 -o air.rgp
```

You are ready to go!


Command Line Options
--------------------
```
optional arguments:
  -f , --fluid          CoolProp fluidname
  -p , --pressures      Pressure range for the RGP table
  -T , --temperatures   Temperature range for the RGP table
  -np , --n_pressures   Number of points to discretize pressure range
  -nT , --n_temperatures 
                        Number of points to discretize temperature range
  -ns , --n_saturation 
                        Number of points to discretize saturation table range
  -o , --output_file    Output file name
```


References
----------

https://github.com/properallan/RGPmeta
