import streamlit as st
import torch
import torch.nn as nn

st.title("PassCDN-PSM口令强度评估器")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stTextInput label{font: normal 1rem courier !important; }
            footer:after {
                content:'Powered By Deadline'; 
                visibility: visible;
                display: block;
                position: relative;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
batch = 64
lstm_hidden = len(chars_table)
lstm_stack_layer = 3
Initial_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9

password=st.text_input("输入测试口令，按回车评估强度:")
password=password.strip()
if len(password) < 1:
    pass
elif len(password) < 6:
    st.write('不支持长度低于6位的口令')
    pass
elif len(password) >= 20:
    st.write('不支持超过20位的口令')
    pass
else:
    max_len=10
    pass_len=len(password)
    sub_passwords=[]
    sub_pass_count=0
    raw_sub_pass=[]
    for length in range(max_len-6+1):
        length = length + 6
        for step in range(pass_len-length+1):
            sub_passwords.append(password[step:length+step])
    ##print(sub_passwords)
    sub_pass_count=len(sub_passwords)
    raw_sub_pass=list(sub_passwords)
    raw_passwords=[]

    for i in range(len(sub_passwords)):
        sub_passwords[i]=sub_passwords[i]+sub_passwords[i]
        sub_passwords[i]=sub_passwords[i]+'`'*(disriminator_max_seq_len-len(sub_passwords[i]))
        data = [char_to_idx[i] for i in sub_passwords[i]]
        data = torch.LongTensor(data)
        raw_passwords.append(data)

    for i in range(batch - len(sub_passwords)):
        data = [char_to_idx[i] for i in '`'*disriminator_max_seq_len]
        data = torch.LongTensor(data)
        raw_passwords.append(data)

    raw_passwords=torch.stack(raw_passwords)

    def idxToOneHot(tensor):
        newT = torch.zeros(tensor.shape[0], tensor.shape[1], chars_table_len)
        for i in range(batch):
            for j in range(disriminator_max_seq_len):
                newT[i][j][tensor[i][j]] = 1
        return newT


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
    discriminator.load_state_dict(torch.load('./discriminator15000'))
    discriminator.eval()

    TPass = idxToOneHot(raw_passwords)
    TPass = torch.reshape(TPass, (batch, disriminator_max_seq_len * chars_table_len))
    TPass = discriminator(TPass)

    scores=TPass.tolist()
    scores = [item for sublist in scores for item in sublist]
    sp_s={}
    for i in range(sub_pass_count):
        sp_s[raw_sub_pass[i]]=scores[i]

    sp_s = dict(sorted(sp_s.items(), key=lambda item: float(item[1]),reverse=True))#collections.OrderedDict(sorted(score_freq.items()))


    finscore=0
    tips=[]

    subp=list(sp_s)
    scores=list(sp_s.values())

    weaksubpass=subp[0]
    score=(scores[0]+scores[1]+scores[2])/3

    nnscore=0
    if score<-1571:
        nnscore=50
    elif score<1244:
        nnscore=40
    elif score<1343:
        nnscore=25
    elif score<1387:
        nnscore=10
    else:
        nnscore=0



    finscore+=nnscore

    rulescore=0

    passlen=len(password)
    if passlen < 8:
        tips.append("口令长度必须大于8位")
    elif passlen >=8 and passlen < 10:
        rulescore+=20
        tips.append("口令长度建议大于10位")
    elif passlen > 10:
        rulescore+=30

    if any(ord(c)>=ord('0') and ord(c)<=ord('9') for c in password):
        rulescore+=5
    else:
        tips.append("口令建议加入数字元素")

    if any(ord(c)>=ord('a') and ord(c)<=ord('z') for c in password) or any(ord(c)>=ord('A') and ord(c)<=ord('Z') for c in password):
        rulescore+=5
    else:
        tips.append("口令建议加入字母元素")

    spchars=r'!@#$%^&*()-=_+[]\{}|,.?'

    spcharscount=0
    for c in password:
        if c in spchars:
            spcharscount+=1
    if spcharscount==0:
        tips.append("口令建议加入至少一位特殊字符")
    elif spcharscount==1:
        rulescore+=5
        tips.append("口令包含2位及以上特殊字符将更安全")
    else:
        rulescore+=10

    if nnscore<40:
        tips.append(f'口令"{password}"中，字符子串"{subp[0]}"为弱口令，请替换')

    finscore+=rulescore


    st.write(f"口令得分:{finscore}")
    if finscore<60:
        st.write(f"口令强度等级:极弱")
    elif finscore<70:
        st.write(f"口令强度等级:弱")
    elif finscore<80:
        st.write(f"口令强度等级:中")
    elif finscore<90:
        st.write(f"口令强度等级:强")
    else:
        st.write(f"口令强度等级:极强")


    for i in tips:
        st.write(i)

    st.write("调试信息:")
    st.write(f"口令规则强度得分:{rulescore}")
    st.write(f"口令在PassCDN得分:{score}")
    st.write(f"口令在PassCDN强度得分:{nnscore}")
    st.write(f'口令子串"{subp[0]}"得分:{scores[0]}')
    st.write(f'口令子串"{subp[1]}"得分:{scores[1]}')
    st.write(f'口令子串"{subp[2]}"得分:{scores[2]}')

    st.write()