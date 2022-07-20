import os
from math import floor

def loadRGP(rgp_name):
    with open(rgp_name, "r") as rgp_fp:
        rgp_data = rgp_fp.read()

    # parse example
    rgp_data = rgp_data.split('$$$$')
    
    data_table = rgp_data[2].split('$TABLE_')
    sat_table = data_table[9].split('$$SAT_TABLE')
    data_table[9] = sat_table[0]
    data_table.append(sat_table[1])
    sat_table = sat_table[1]
       
    #print('-')
    #print(data_table[9])
    #print(data_table[10].replace('\n','').split())
    data_table = [data.replace('\n', '').split() for data in data_table[1:len(data_table)]]
    
    #print(data_table[8])

    for i in range(0, len(data_table)):
        data = data_table[i][3:len(data_table[i])]
        if i == len(data_table)-1:
            nPoints = int(data_table[i][0])
        else:
            nPoints = int(data_table[i][2])
        nPoints = 5
        data_table[i] = []
        for j in range(0, floor(len(data)/nPoints)):
            data_table[i].append(data[j*nPoints:(j+1)*nPoints])

    return data_table

#example_rgp = loadRGP('fourthstage_640K61MPa_1200x1200.rgp')
#example_rgp = loadRGP('5x5_1.rgp')
example_rgp = loadRGP('format_example.rgp')


print('-----------------------------------------')
#our_rgp = loadRGP('co2_1200x1200.rgp')
#our_rgp = loadRGP('5x5_3.rgp')
our_rgp = loadRGP('R134a_5x5.rgp')


for i in range(0, len(example_rgp)):
    table_ex = example_rgp[i]
    table_ou = our_rgp[i]
    

    for j in range(0, len(table_ex)):
        
        phi_ex = [float(num) for num in table_ex[j]]
        phi_ou = [float(num) for num in table_ou[j]]

        for k in range(0, len(phi_ex)):
            
            diff = 0
                
            diff = (phi_ex[k] - phi_ou[k]) / phi_ex[k]
            
            if diff>1e-2:
                print('Table', i+1, 'phi', j+1, 'item', k+1, 'diff = ', round(diff,4),
                      'ex', round(phi_ex[k],4), 'ou', round(phi_ou[k],4))

print('ok')