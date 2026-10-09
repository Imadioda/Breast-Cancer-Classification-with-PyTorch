import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

dtf=pd.read_csv("BreastCancer.csv",sep=",",decimal=".")

dtf = dtf.drop(columns="id")
dtf = dtf.drop(dtf.columns[-1],axis=1)


x = dtf.iloc[:,0]
x=x.map({"M":1,"B":0})
print(x.head)


dtf_calc = dtf.drop(dtf.columns[0],axis=1)
#print(dtf_calc.isna().sum())

normalizer = StandardScaler()
dtf_norm = pd.DataFrame(normalizer.fit_transform(dtf_calc),columns=dtf_calc.columns,index=dtf_calc.index)

#print(np.var(dtf_norm,axis=0))
cor=dtf_norm.corr().abs()
var_asupp=[]

for i in range(len(cor.columns)):
    for j in range(i):
        if cor.iloc[i,j]>0.9:
            var_asupp.append(cor.columns[i])

#print(var_asupp)

dtf_sans_cor=dtf_norm.drop(columns=var_asupp)

print(dtf_sans_cor.info())

var=torch.tensor(x.to_numpy(), dtype=torch.float32)
data=torch.tensor(dtf_sans_cor.to_numpy(), dtype=torch.float32)
dataset=TensorDataset(data,var)

