import torch
from torch.utils.data import DataLoader,TensorDataset
from DatasetClean import dataset
import torch.nn as nn
import copy

#print(dataset.tensors[0].shape)
#print(dataset.tensors[1].shape)

class MonReseau(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.couche1=nn.Linear(20, 10)
        self.relu=nn.ReLU()
        self.batchnorm=nn.BatchNorm1d(10)
        self.dropout=nn.Dropout(0.3)
        self.couche2=nn.Linear(10, 1)

    def forward(self,x):
        x=self.couche1(x)
        x=self.relu(x)
        x=self.batchnorm(x)
        x=self.dropout(x)
        x=self.couche2(x)
        return x

model=MonReseau()

variables=dataset[0]
x=dataset[1]

torch.manual_seed(42)
idx=torch.randperm(len(dataset))

data=dataset.tensors[0]
var=dataset.tensors[1]

train_idx=idx[:int(len(dataset)*0.7)]
validation_idx=idx[int(len(dataset)*0.7):int(len(dataset)*0.85)]
test_idx=idx[int(len(dataset)*0.85):]

train=TensorDataset(data[train_idx],var[train_idx])
validation=TensorDataset(data[validation_idx],var[validation_idx])
test=TensorDataset(data[test_idx],var[test_idx])


databatch=DataLoader(dataset=train,batch_size=32,shuffle=True)
loss_function=nn.BCEWithLogitsLoss()
epochs=50
optimisation=torch.optim.SGD(model.parameters(),lr=0.01)

best_loss=float("inf")
compteur=0
max=6

for epoch in range(epochs):
    sum_loss=0.0
    loss_train_mean=0.0
    loss_validation=0.0

    model.train()
    for i,j in databatch:
        optimisation.zero_grad()

        train_pred=model(i)
        loss=loss_function(train_pred,j.unsqueeze(1))

        loss.backward()

        sum_loss+=loss.item()
        loss_train_mean=sum_loss/len(databatch)

        optimisation.step()

    model.eval()

    with torch.no_grad():
        logit_validation=model(validation.tensors[0])
        lossv=loss_function(logit_validation,validation.tensors[1].unsqueeze(1))
        loss_validation=lossv.item()

        validation_pred=torch.sigmoid(logit_validation)
        fv_pred=(validation_pred>=0.5).float()

        accuracy_validation=(fv_pred==validation.tensors[1].unsqueeze(1)).float().mean()

        print(f"Epoch : {epoch+1}, Loss moyenne : {loss_train_mean}, Loss validation : {loss_validation}, Accuracy validation : {accuracy_validation}")

        if loss_validation<best_loss:
            best_loss=loss_validation
            compteur=0
            model_sv=copy.deepcopy(model)

        else :
            compteur+=1
            if compteur==max:
                torch.save(model_sv.state_dict(), "model.pth")
                print("Early stopping, model has been saved ")
                break

model=model_sv
torch.save(model.state_dict(), "model.pth")

with torch.no_grad():
    logit_test=model(test.tensors[0])
    pred_test=torch.sigmoid(logit_test)
    ft_pred=(pred_test>=0.5).float()

    test_accuracy=(ft_pred==test.tensors[1].unsqueeze(1)).float().mean()
    print(f"Test accuracy : {test_accuracy}")