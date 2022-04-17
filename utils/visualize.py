import json
# import matplotlib as plt

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['font.sans-serif'] = ['Times New Roman']


dataset_name = 'cspider'
if dataset_name == 'dusql':
    num_epochs = 80
elif dataset_name == 'cspider':
    num_epochs = 266
elif dataset_name == 'nl2sql':
    num_epochs = 44

original = f'{dataset_name}/{dataset_name}-nll/log'
BCE = f'{dataset_name}/{dataset_name}-bce/log'

train_loss = []
val_loss = []
val_acc = []

bce_train_loss = []
bce_val_loss = []
bce_val_acc = []

for x in range(num_epochs):
    with open(f'{original}/metrics_epoch_{x}.json', 'r') as f:
        temp = json.load(f)
        train_loss.append(temp['training_loss'])
        val_loss.append(temp['validation_loss'])
        val_acc.append(temp['validation_spider'])

for x in range(num_epochs):
    with open(f'{BCE}/metrics_epoch_{x}.json', 'r') as f:
        temp = json.load(f)
        bce_train_loss.append(temp['training_loss'])
        bce_val_loss.append(temp['validation_loss'])
        bce_val_acc.append(temp['validation_spider'])

x_data = range(num_epochs)


fig = plt.figure()

ax = fig.add_subplot(111)

lns1 = ax.plot(x_data, train_loss, color='black', linewidth=2.0, linestyle='-', label = 'NLLLoss')
ax2 = ax.twinx()
lns2 = ax2.plot(x_data, bce_train_loss, color='black', linewidth=2.0, linestyle='--', label = 'BCELoss')

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('NLLLoss', fontsize=20)
ax2.set_ylabel('BCELoss', fontsize=20)
plt.tight_layout()
plt.savefig(f'train_loss_{dataset_name}.pdf', format='pdf')
plt.savefig(f'train_loss_{dataset_name}.png')
plt.clf()

original_, = plt.plot(x_data, val_acc, color='black', linewidth=2.0, linestyle='-')

bce_, = plt.plot(x_data, bce_val_acc, color='black', linewidth=2.0, linestyle='--')

plt.legend(handles=[original_, bce_], labels=['NLLLoss', 'BCELoss'])
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Validation Accuracy', fontsize=20)
plt.tight_layout()
plt.savefig(f'val_acc_{dataset_name}.pdf', format='pdf')
plt.savefig(f'val_acc_{dataset_name}.png')
plt.clf()