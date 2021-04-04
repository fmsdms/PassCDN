import torch
import torch.nn as nn
import mmap
import sys
import os
import torch.utils.data as data_utils

# char table to one hot
chars_table = '`qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890~!@#$%^&*()_+-=[]{};\'":,.<>/?|\\'
chars_table_one_hot_from_char = {}
chars_table_one_hot_from_idx = []
chars_table_len = len(chars_table)
for i in range(chars_table_len):
    init_list = [0 for j in range(chars_table_len)]
    init_list[i] = 1
    chars_table_one_hot_from_char[chars_table[i]] = init_list
    chars_table_one_hot_from_idx.append(init_list)

char_to_idx = {chars_table[i]: i for i in range(chars_table_len)}
idx_to_char = chars_table
# end char table 2 one hot

# parameters from generator
noise_input_dim = 128
generate_max_seq_len = 10
lstm_input_size = 16

# parameters from disminator
disriminator_max_seq_len = 21
embed_dim = 16

# public parameters
batch = 128
lstm_hidden = len(chars_table)
lstm_stack_layer = 3
Initial_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

password_file=r'C:\Users\long\Desktop\data_longlong\n5m1e7.txt'
password_test_file=r'C:\Users\long\Desktop\data_longlong\rockyou-test.txt'

passwords=[]
raw_passwords=[]
testpass=[]
raw_testpass=[]
with open(password_file, 'rb') as f:
    # m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) #File is open read-only
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    data = m.readline()
    while data:
        try:
            data = data.decode("utf-8").strip()
            if len(data) > generate_max_seq_len:
                # print(f'{data}\'s length > 10')
                continue
            try:
                # data = passwords2embeds(data)
                raw_passwords.append(data)
                data = data + data + '`' * (disriminator_max_seq_len - len(data) * 2)
                data = [char_to_idx[i] for i in data]
                data = torch.LongTensor(data)
                passwords.append(data)
                data = m.readline()
                continue
            except:
                # print(f'{data} contains invalid charactors')
                data = m.readline()
                continue
        except:
            # print(f'{data} cannot be decoded')
            data = m.readline()
            continue
with open(password_test_file, 'rb') as f:
    # m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) #File is open read-only
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    data = m.readline()
    while data:
        try:
            data = data.decode("utf-8").strip()
            if len(data) > generate_max_seq_len:
                # print(f'{data}\'s length > 10')
                continue
            try:
                raw_testpass.append(data)
                # data = passwords2embeds(data)
                data = data + data + '`' * (disriminator_max_seq_len - len(data) * 2)
                data = [char_to_idx[i] for i in data]
                data = torch.LongTensor(data)
                testpass.append(data)
                data = m.readline()
                continue
            except:
                # print(f'{data} contains invalid charactors')
                data = m.readline()
                continue
        except:
            # print(f'{data} cannot be decoded')
            data = m.readline()
            continue

def idxToOneHot(tensor):
    newT = torch.zeros(tensor.shape[0], tensor.shape[1], chars_table_len)
    for i in range(batch):
        for j in range(disriminator_max_seq_len):
            newT[i][j][tensor[i][j]] = 1
    return newT.cuda()

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.embed = nn.Linear(disriminator_max_seq_len * chars_table_len, disriminator_max_seq_len * embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=lstm_hidden, num_layers=lstm_stack_layer,
                            batch_first=True,
                            dropout=0.1)
        self.LRelu = nn.LeakyReLU(0.01)
        self.score = nn.Linear(disriminator_max_seq_len * lstm_hidden, 1)

    def forward(self, input):
        output = self.embed(input)
        output = torch.reshape(output, (batch, disriminator_max_seq_len, embed_dim))
        output, _ = self.lstm(output)
        output = self.LRelu(output)
        output = torch.reshape(output, (batch, disriminator_max_seq_len * lstm_hidden))
        output = self.score(output)
        return output

discriminator=Discriminator()
discriminator.load_state_dict(torch.load('./model/rockyou-train/discriminator77000'))
discriminator.cuda().eval()

sys.stdout=open(f'hits.log','w')
sys.stderr=sys.stdout

alllen=len(passwords)

index=0
while index<alllen:
    batchpass=passwords[index:index+batch]

    batchpass=torch.stack(batchpass)
    TPass = idxToOneHot(batchpass)
    TPass = torch.reshape(TPass, (batch, disriminator_max_seq_len * chars_table_len))
    TPass = discriminator(TPass)
    scores = TPass.tolist()
    for i in range(batch):
        if raw_passwords[index:index+batch][i] in raw_testpass:
            print(f"{str(scores[i][0])} {'1'}")
        else:
            print(f"{str(scores[i][0])} {'0'}")
    index += batch
    sys.stdout.flush()



