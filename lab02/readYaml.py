import numpy as np 
from ruamel import yaml
from decimal import Decimal

class ReadYaml:

    def __init__(self,path):
        
        with open(path, 'r', encoding='utf-8') as doc:
           
               self.content = yaml.load(doc, Loader=yaml.Loader)
            #   print(self.content['value'])


    def readTonumpyArray(self):
        
        txt = self.content['value']

        txt_list = txt.split('{')
        
        
       # print(txt_list)
        self.homo = []
        
        for idx ,ele in enumerate(txt_list):
            
           if idx == 0:
               continue
           
           row = []
        #   print('**************')
        #   print(ele[3:].split(','))
           row_txt = ele[3:].split(',')
           self.homo.append(row)
           for col in  row_txt:
                                             
                val = float(col)
            #    print(val)
                row.append(val)

        #   print(self.homo)
        
        return np.array(self.homo)

#homo  = ReadYaml('config.yaml')
#print('#############################')
#homo_mat = homo.readTonumpyArray()
#
#print(homo_mat)

#print(type(homo_mat))

