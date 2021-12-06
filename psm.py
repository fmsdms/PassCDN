import streamlit as st
import torch
import torch.nn as nn


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
def app():
    st.title("PassCDN Password Strength Meter")
    st.write('-------------------------')
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

    disriminator_max_seq_len = 21
    embed_dim = 16

    # public parameters
    lstm_hidden = len(chars_table)
    lstm_stack_layer = 3

    password=st.text_input("Input your password,press Enter to check your password's strength")
    password=password.strip()
    bar = st.progress(0)
    if len(password) < 1:
        pass
    elif len(password) < 6:
        st.write('Passwords whose length is shorter than 6 are not supported')
        pass
    elif len(password) > 30:
        st.write('Passwords whose length is longer than 30 are not supported')
        pass
    else:
        min_len=4
        max_len=10
        pass_len=len(password)
        sub_passwords=[]
        for length in range(max_len-min_len+1):
            length = length + min_len
            for step in range(pass_len-length+1):
                sub_passwords.append(password[step:length+step])

        sub_pass_count=len(sub_passwords)
        raw_sub_pass=list(sub_passwords)
        raw_passwords=[]

        for i in range(len(sub_passwords)):
            sub_passwords[i]=sub_passwords[i]+sub_passwords[i]
            sub_passwords[i]=sub_passwords[i]+'`'*(disriminator_max_seq_len-len(sub_passwords[i]))
            data = [char_to_idx[i] for i in sub_passwords[i]]
            data = torch.LongTensor(data)
            raw_passwords.append(data)

        # for i in range(batch - len(sub_passwords)):
        #     data = [char_to_idx[i] for i in '`'*disriminator_max_seq_len]
        #     data = torch.LongTensor(data)
        #     raw_passwords.append(data)

        raw_passwords=torch.stack(raw_passwords)

        all_sub_pass_count=len(raw_passwords)

        def idxToOneHot(tensor,count):
            newT = torch.zeros(tensor.shape[0], tensor.shape[1], chars_table_len)
            for i in range(count):
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
                output = torch.reshape(output, (all_sub_pass_count, disriminator_max_seq_len, embed_dim))
                output, _ = self.lstm(output)
                output = self.LRelu(output)
                output = torch.reshape(output, (all_sub_pass_count, disriminator_max_seq_len * lstm_hidden))
                output = self.score(output)
                return output

        discriminator=Discriminator()
        discriminator.load_state_dict(torch.load('./discriminator15000',map_location=torch.device('cpu')))
        discriminator.eval()

        TPass = idxToOneHot(raw_passwords,all_sub_pass_count)
        TPass = torch.reshape(TPass, (all_sub_pass_count, disriminator_max_seq_len * chars_table_len))
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

        score=scores[0]*0.7+scores[1]*0.2+scores[2]*0.1

        nnscore=0
        if score<-1560:
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
            tips.append("Try improving by adding characters")
        elif passlen >=8 and passlen < 10:
            rulescore+=20
            tips.append("Passwords whose length is longer than 10 will have more credits")
        elif passlen > 10:
            rulescore+=30

        if any(ord(c)>=ord('0') and ord(c)<=ord('9') for c in password):
            rulescore+=5
        else:
            tips.append("Try improving by adding numbers")

        if any(ord(c)>=ord('a') and ord(c)<=ord('z') for c in password) or any(ord(c)>=ord('A') and ord(c)<=ord('Z') for c in password):
            rulescore+=5
        else:
            tips.append("Try improving by adding alphabets")

        spchars=r'!@#$%^&*()-=_+[]\{}|,.?'

        spcharscount=0
        for c in password:
            if c in spchars:
                spcharscount+=1
        if spcharscount==0:
            tips.append("Try improving by adding special characters")
        elif spcharscount==1:
            rulescore+=5
            tips.append("Paswords whose length contains 2 more special characters will have more credits")
        else:
            rulescore+=10

        if nnscore<40:
            tips.append(f'The substring "{subp[0]}" of "{password}" is a vulnerable component,try replacing that.')

        finscore+=rulescore

        bar.progress(finscore)

        st.write(f"Password strength score:{finscore}")
        if finscore<60:
            st.write(f"Password strength level: Worst")
        elif finscore<70:
            st.write(f"Password strength level: Weak")
        elif finscore<80:
            st.write(f"Password strength level: Common")
        elif finscore<90:
            st.write(f"Password strength level: Strong")
        else:
            st.write(f"Password strength level: Best")


        for i in tips:
            st.write(i)

        st.write("Debug Info:")
        st.write(f"Scoring by Rule: {rulescore}")
        st.write(f"Scoring by PassCDN: {nnscore}")
        st.write(f"Output by PassCDN: {score}")
        st.write(f'Score of weak top 1 substring"{subp[0]}": {scores[0]}')
        st.write(f'Score of weak top 2 substring"{subp[1]}": {scores[1]}')
        st.write(f'Score of weak top 3 substring"{subp[2]}": {scores[2]}')
