import time
import torch

# torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(int(time.time()))
import random
from torch import nn
import mmap
import os
import sys
import torch.utils.data as data_utils

torch.autograd.set_detect_anomaly(True)

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


# CDN
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


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear = nn.Linear(noise_input_dim, generate_max_seq_len * lstm_input_size)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=lstm_hidden, num_layers=lstm_stack_layer,
                            batch_first=True, dropout=0.1)

    def forward(self, lastBatch=None):
        noise = torch.randn(size=(batch, noise_input_dim)).cuda()
        output = self.linear(noise)
        output = torch.reshape(output, (batch, generate_max_seq_len, lstm_input_size))
        output, _ = self.lstm(output)
        output = torch.softmax(output, 2)
        if lastBatch is not None:
            different = lastBatch - output
            return output, different
        return output


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


generator = Generator().cuda().train()
discriminator = Discriminator().cuda().train()

optG = torch.optim.Adam(generator.parameters(), lr=Initial_learning_rate, betas=(beta1, beta2))
optD = torch.optim.Adam(discriminator.parameters(), lr=Initial_learning_rate, betas=(beta1, beta2))
for param in generator.parameters():
    param.requires_grad = True
for param in discriminator.parameters():
    param.requires_grad = True
# end CDN

# deal with file

password_file = r'C:\Users\long\Desktop\data_longlong\rockyou-train.txt'
dataset_name=password_file.split('\\')[-1].split('.')[0]
# password_test_file=r'C:\Users\long\Desktop\data_longlong\rockyou-test.txt'
raw_passwords = []
training_set_rate = 0.6
validating_set_rate = 0.4

sys.stdout=open(f'{dataset_name}_CDN.log','w')
sys.stderr=sys.stdout

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
                data = data + data + '`' * (disriminator_max_seq_len - len(data) * 2)
                data = [char_to_idx[i] for i in data]
                data = torch.LongTensor(data)
                raw_passwords.append(data)
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

random.shuffle(raw_passwords)
training_passwords = raw_passwords[:int(len(raw_passwords) * training_set_rate)]
validating_passwords = raw_passwords[int(len(raw_passwords) * training_set_rate):int(
    len(raw_passwords) * training_set_rate + len(raw_passwords) * validating_set_rate)]
training_dataloader = data_utils.DataLoader(training_passwords, batch_size=batch)
validating_dataloader = data_utils.DataLoader(validating_passwords, batch_size=batch)


def get_batch_training():
    return next(iter(training_dataloader))


def get_batch_validating():
    return next(iter(validating_dataloader))


# end deal with file
# train
max_iter = 400000
length_freq_list = [5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 10]

indices5 = torch.LongTensor([0, 1, 2, 3, 4]).cuda()  # construct gather index
indices5 = torch.unsqueeze(indices5, 0)
indices5 = torch.cat([indices5] * batch)
indices6 = torch.LongTensor([0, 1, 2, 3, 4, 5]).cuda()
indices6 = torch.unsqueeze(indices6, 0)
indices6 = torch.cat([indices6] * batch)
indices7 = torch.LongTensor([0, 1, 2, 3, 4, 5, 6]).cuda()
indices7 = torch.unsqueeze(indices7, 0)
indices7 = torch.cat([indices7] * batch)
indices8 = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
indices8 = torch.unsqueeze(indices8, 0)
indices8 = torch.cat([indices8] * batch)
indices9 = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8]).cuda()
indices9 = torch.unsqueeze(indices9, 0)
indices9 = torch.cat([indices9] * batch)
indices10 = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).cuda()
indices10 = torch.unsqueeze(indices10, 0)
indices10 = torch.cat([indices10] * batch)
indices = [0, 0, 0, 0, 0, indices5, indices6, indices7, indices8, indices9, indices10]

C0=0.1
C1=0.9
C2=0.9
A=1.1
everyIter=2000

def idxToOneHot(tensor):
    newT = torch.zeros(tensor.shape[0], tensor.shape[1], chars_table_len)
    for i in range(batch):
        for j in range(disriminator_max_seq_len):
            newT[i][j][tensor[i][j]] = 1

    return newT.cuda()


for iter_ in range(max_iter):
    lastGenBatch = generator()  # init
    # update D
    optD.zero_grad()
    Gpass = generator()
    Gpass = torch.argmax(Gpass, 2).long()
    random_length = length_freq_list[random.randint(0, len(length_freq_list) - 1)]
    Gpass = torch.gather(Gpass, 1, indices[random_length])
    ending = torch.zeros((batch, disriminator_max_seq_len - random_length * 2)).long().cuda()
    Gpass = torch.cat((Gpass, Gpass, ending), dim=1)
    Gpass = idxToOneHot(Gpass)
    Gpass = torch.reshape(Gpass, (batch, disriminator_max_seq_len * chars_table_len))
    Gpass_score = discriminator(Gpass)

    Dpass = get_batch_training()
    Dpass = idxToOneHot(Dpass)
    Dpass = torch.reshape(Dpass, (batch, disriminator_max_seq_len * chars_table_len))
    Dpass_score = discriminator(Dpass)

    lossD = C0 / (A ** (iter_ / everyIter)) * torch.mean(Gpass_score) - C1 * torch.mean(Dpass_score)
    lossD.backward(retain_graph=True)

    optD.step()

    # update G
    optG.zero_grad()
    Gpass, diff = generator(lastGenBatch)
    lastGenBatch = torch.cat((Gpass, lastGenBatch), dim=0)
    lastGenBatch = lastGenBatch[torch.randperm(lastGenBatch.size()[0])].view(lastGenBatch.size())
    lastGenBatch = lastGenBatch[:batch]
    Gpass = torch.argmax(Gpass, 2).long()
    random_length = length_freq_list[random.randint(0, len(length_freq_list) - 1)]
    Gpass = torch.gather(Gpass, 1, indices[random_length])
    ending = torch.zeros((batch, disriminator_max_seq_len - random_length * 2)).long().cuda()
    Gpass = torch.cat((Gpass, Gpass, ending), dim=1)
    Gpass = idxToOneHot(Gpass)
    Gpass = torch.reshape(Gpass, (batch, disriminator_max_seq_len * chars_table_len))
    Gpass_score = discriminator(Gpass)

    lossG = C0 / (A ** (iter_ / everyIter)) * torch.mean(Gpass_score) - C2 * torch.mean(torch.abs(diff))
    lossG.backward(retain_graph=True)

    optG.step()

    Vpass = get_batch_validating()
    Vpass = idxToOneHot(Vpass)
    Vpass = torch.reshape(Vpass, (batch, disriminator_max_seq_len * chars_table_len))
    Vpass_score = discriminator(Vpass)

    if iter_ % 50 == 0:
        if not os.path.exists(f'./model/{dataset_name}'):
            os.makedirs(f'./model/{dataset_name}')
        torch.save(generator.state_dict(), f'./model/{dataset_name}/generator{iter_}')
        torch.save(discriminator.state_dict(), f'./model/{dataset_name}/discriminator{iter_}')

    print(
        f"iter:{iter_}\tlossD:{lossD}\tlossG:{lossG}\tnot_pass_judge:{torch.mean(Gpass_score)}\tis_pass_judge:{torch.mean(Dpass_score)}\tdiff:{torch.mean(torch.abs(diff))}\tjudge_from_validating:{torch.mean(Vpass_score)}")

    sys.stdout.flush()

# end train