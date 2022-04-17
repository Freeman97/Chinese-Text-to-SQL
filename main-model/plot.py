import json
# import matplotlib as plt

import matplotlib.pyplot as plt
num_epochs = 44

original = 'experiments/nl2sql-T-7-K-30-redo4'
BCE = 'experiments/nl2sql-T-7-K-30-BCE-redo'

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

lns1 = ax.plot(x_data, train_loss, color='black', linestyle='-', label = 'NLLLoss')
ax2 = ax.twinx()
lns2 = ax2.plot(x_data, bce_train_loss, color='black', linestyle='-.', label = 'BCELoss')

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs)
ax.set_xlabel('Epoch')
ax.set_ylabel('NLLLoss')
ax2.set_ylabel('BCELoss')
plt.tight_layout()
plt.savefig('train_loss_nl2sql.pdf', format='pdf')
plt.clf()

# original, = plt.plot(x_data, val_loss, color='black', linestyle='-')
# bce, = plt.plot(x_data, bce_val_loss, color='black', linestyle='--')
# plt.legend(handles=[original, bce], labels=['NLLLoss', 'BCELoss'])
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.savefig('val_loss.png')
# plt.clf()

original_, = plt.plot(x_data, val_acc, color='black', linestyle='-')
bce_, = plt.plot(x_data, bce_val_acc, color='black', linestyle='-.')
plt.legend(handles=[original_, bce_], labels=['NLLLoss', 'BCELoss'])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.savefig('val_calc_nl2sql.pdf', format='pdf')
plt.clf()