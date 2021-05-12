import pickle
import random
import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import spacy
nlp = spacy.load('en_core_web_sm')

use_cuda = torch.cuda.is_available()
print("use_cuda: {}".format(use_cuda))

class StackDataset(Dataset):
    def __init__(self, p_dataset=None, 
                 p_tmaxlen=None,
                 p_wmaxlen=None, 
                 p_hmaxlen=None, # should be 40 by default
                 p_fmaxlen=None, 
                 p_stack=None):
        if p_stack is None:
            self.word_list = p_dataset["word_list"]
            self.word_dict = p_dataset["word_dict"]
            self.hint_list = p_dataset["hint_list"]
            self.hint_dict = p_dataset["hint_dict"]
            self.func_list = p_dataset["func_list"]
            self.func_dict = p_dataset["func_dict"]
            self.tokenized_data = p_dataset["tokenized_data"]
            self.Nr = len(p_dataset["tokenized_data"])
            self.tNc = p_tmaxlen
            self.wNc = p_wmaxlen
            self.hNc = p_hmaxlen
            self.fNc = p_fmaxlen
            self.InitializeData()
            # self.FLengthFilter(5,9)
        else:
            self.init_by_slice(p_stack=p_stack)
        
    def init_by_slice(self, p_stack=None):
        self.tMTX = p_stack[0]
        self.wMTX = p_stack[1]
        self.hMTX = p_stack[2] # ####
        self.fMTX = p_stack[3]
        self.tLEN = p_stack[4]
        self.wLEN = p_stack[5]
        self.hLEN = p_stack[6]
        self.fLEN = p_stack[7]
        assert self.tMTX.shape[0]==self.wMTX.shape[0]==self.fMTX.shape[0]==self.hMTX.shape[0]
        assert self.tLEN.shape[0]==self.wLEN.shape[0]==self.fLEN.shape[0]==self.hLEN.shape[0]
        self.Nr = self.wMTX.shape[0]
        self.tNc = self.tMTX.shape[1]
        self.wNc = self.wMTX.shape[1]
        self.hNc = self.hMTX.shape[1]
        self.fNc = self.fMTX.shape[1]
    
    def __len__(self):
        return self.Nr
    
    def __getitem__(self, p_ind):
        return (
            self.tMTX[p_ind, :],
            self.wMTX[p_ind, :],
            self.hMTX[p_ind, :], # ####
            self.fMTX[p_ind, :],
            self.tLEN[p_ind],
            self.wLEN[p_ind],
            self.hLEN[p_ind],
            self.fLEN[p_ind],
        )
    
    def InitializeData(self):
        self.tMTX = np.full((self.Nr, self.tNc), self.word_dict["<PAD>"], dtype=int)
        self.wMTX = np.full((self.Nr, self.wNc), self.word_dict["<PAD>"], dtype=int)
        self.hMTX = np.full((self.Nr, self.hNc), self.hint_dict["<PAD>"], dtype=int)
        self.fMTX = np.full((self.Nr, self.fNc), self.func_dict["<PAD>"], dtype=int)
        self.tLEN = np.full((self.Nr,), -1, dtype=int)
        self.wLEN = np.full((self.Nr,), -1, dtype=int)
        self.hLEN = np.full((self.Nr,), -1, dtype=int)
        self.fLEN = np.full((self.Nr,), -1, dtype=int)
        # start filling
        for i in range(self.Nr):
            print("\r# Filling {}/{}".format(i,self.Nr),end="")
            d_titl = self.tokenized_data[i][0]
            d_ques = self.tokenized_data[i][1]
            d_hint = self.tokenized_data[i][2]
            d_ansr = self.tokenized_data[i][3]
            for j in range(min(len(d_titl),self.tNc)):
                self.tMTX[i,j] = d_titl[j]
            for j in range(min(len(d_ques),self.wNc)):
                self.wMTX[i,j] = d_ques[j]
            for j in range(min(len(d_hint),self.hNc)):
                self.hMTX[i,j] = d_hint[j]
            for j in range(min(len(d_ansr),self.fNc)):
                self.fMTX[i,j] = d_ansr[j]
            self.tLEN[i] = min(len(d_titl),self.tNc)
            self.wLEN[i] = min(len(d_ques),self.wNc)
            self.hLEN[i] = min(len(d_hint),self.hNc)
            self.fLEN[i] = min(len(d_ansr),self.fNc)
        print()
        print("# Done Filling.")
            
    def FLengthFilter(self, p_minlen, p_maxlen):
        # filter out those under minlen
        d_valid_idx = self.fLEN>=p_minlen
        self.tMTX = self.tMTX[d_valid_idx,:]
        self.wMTX = self.wMTX[d_valid_idx,:]
        self.hMTX = self.hMTX[d_valid_idx,:]
        self.fMTX = self.fMTX[d_valid_idx,:]
        self.tLEN = self.tLEN[d_valid_idx]
        self.wLEN = self.wLEN[d_valid_idx]
        self.hLEN = self.hLEN[d_valid_idx]
        self.fLEN = self.fLEN[d_valid_idx]
        d_valid_idx = self.fLEN<=p_maxlen
        self.tMTX = self.tMTX[d_valid_idx,:]
        self.wMTX = self.wMTX[d_valid_idx,:]
        self.hMTX = self.hMTX[d_valid_idx,:]
        self.fMTX = self.fMTX[d_valid_idx,:]
        self.tLEN = self.tLEN[d_valid_idx]
        self.wLEN = self.wLEN[d_valid_idx]
        self.hLEN = self.hLEN[d_valid_idx]
        self.fLEN = self.fLEN[d_valid_idx]
        self.Nr = self.wMTX.shape[0]
        self.tNc = self.tMTX.shape[1]
        self.wNc = self.wMTX.shape[1]
        self.hNc = self.hMTX.shape[1]
        self.fNc = self.fMTX.shape[1]
        print("# Filtered, {} preserved.".format(self.Nr))

class Simple_Encoder_LSTM(nn.Module):
    def __init__(self, p_vocab_size=None):
        super(Simple_Encoder_LSTM, self).__init__()
        self.IDX_PAD = 0
        self.IDX_SOS = 1
        self.IDX_EOS = 2
        self.IDX_UKN = 3
        self.vocab_size = p_vocab_size
        self.hidden_size = 256
        self.embd_dim = 256
        self.embedding = nn.Embedding(
            p_vocab_size,
            self.embd_dim,
            padding_idx=self.IDX_PAD
        )
        self.lstm = nn.LSTM(
            self.embd_dim,
            self.hidden_size,
            num_layers=1,
        )
    
    def move_to_gpu(self):
        self.embedding = self.embedding.cuda()
        self.lstm = self.lstm.cuda()
        
    def forward(self, p_seq, p_len):
        # p_seq: (B, maxlen)
        B = p_seq.shape[0]
        D = p_seq.shape[1]
        tmp1 = self.embedding(p_seq).transpose(0,1) # (B, maxlen, embd_dim) -> (maxlen, B, embd_dim)
        
        # try to use a more simple test code?
        # tmp5, (h5, c5) = self.lstm(tmp1)
        
        s_lens, perm_idx = p_len.sort(0, descending=True)
        tmp1 = tmp1[:,perm_idx,:]
        tmp2 = pack_padded_sequence(tmp1, s_lens.cpu().numpy())
        tmp3, (h3, c3) = self.lstm(tmp2) # h3/c3: (bi*num_layers, B, hidden)
        tmp4, _ = pad_packed_sequence(tmp3, total_length=D) # tmp4: (maxlen, B, bi*hidden)
        # re-order
        _, unperm_idx = perm_idx.sort(0)
        tmp5 = tmp4[:,unperm_idx,:]
        h5 = h3[:,unperm_idx,:]
        c5 = c3[:,unperm_idx,:]
        
        return (h5,c5)

class Double_Decoder_LSTM(nn.Module):
    def __init__(self, p_vocab_size=None):
        super(Double_Decoder_LSTM, self).__init__()
        self.IDX_PAD = 0
        self.IDX_SOS = 1
        self.IDX_EOS = 2
        self.IDX_UKN = 3
        self.hidden_size = 256
        self.embd_dim = 256
        self.embedding = nn.Embedding(
            p_vocab_size,
            self.embd_dim,
            padding_idx=self.IDX_PAD,
        )
        self.lstm1 = nn.LSTM(
            self.embd_dim, # input size
            self.hidden_size, # hidden size
            num_layers=1,
        )
        self.lstm2 = nn.LSTM(
            self.embd_dim, # input size
            self.hidden_size, # hidden size
            num_layers=1,
        )
        self.linear = nn.Linear(self.hidden_size*2, p_vocab_size) # #### gen5: cat two outputs #### #
    
    def move_to_gpu(self):
        self.embedding = self.embedding.cuda()
        self.lstm1 = self.lstm1.cuda()
        self.lstm2 = self.lstm2.cuda()
        self.linear = self.linear.cuda()
    
    def forward(self, p_seq, p_len, p_hiddens1, p_hiddens2):
        # ### using ignore_index in NLLLoss to ignore 0 padding output ### #
        s_seq = p_seq[:,:-1] # trim
        tmp1 = self.embedding(s_seq).transpose(0,1) # (B, maxlen-1, embd_dim) -> (maxlen-1, B, embd_dim)
        output1, (hidden1, cell1) = self.lstm1(tmp1, p_hiddens1) # output: (maxlen-1, B, bi*hidden)
        output2, (hidden2, cell2) = self.lstm2(tmp1, p_hiddens2)
        tmp2 = output1.transpose(0,1) # tmp2: (B, maxlen-1, bi*hidden)
        tmp3 = output2.transpose(0,1)
        tmp4 = torch.cat([tmp2,tmp3], dim=2)
        prediction = self.linear(tmp4) # prediction: (B, maxlen-1, vocab_size)
        return prediction

class DoubleSeq2Seq(nn.Module):
    def __init__(self,
                 p_titl_vocab_size=None,
                 p_word_vocab_size=None,
                 p_func_vocab_size=None,
                ):
        super(DoubleSeq2Seq, self).__init__()
        self.w_encoder = Simple_Encoder_LSTM(
            p_vocab_size = p_word_vocab_size,
        )
        self.t_encoder = Simple_Encoder_LSTM(
            p_vocab_size = p_titl_vocab_size,
        )
        self.f_decoder = Double_Decoder_LSTM(
            p_vocab_size = p_func_vocab_size,
        )
    
    def move_to_gpu(self):
        self.t_encoder.move_to_gpu()
        self.w_encoder.move_to_gpu()
        self.f_decoder.move_to_gpu()
    
    def forward(self, p_tseq, p_wseq, p_fseq, p_tlen, p_wlen, p_flen):
        t_hidden = self.t_encoder(p_tseq, p_tlen)
        w_hidden = self.w_encoder(p_wseq, p_wlen) # (h,c)
        tmp2 = self.f_decoder(p_fseq, p_flen, t_hidden, w_hidden) # (B, maxlen, hidden)
        return tmp2
        
    def score(self, p_tseq, p_wseq, p_fseq, p_tlen, p_wlen, p_flen):
        # given target function sequence and description
        # return the log likelihood for the sequence
        # p_output is like <SOS>,A,B,..., with no <EOS>
        t_hidden = self.t_encoder(p_tseq, p_tlen)
        w_hidden = self.w_encoder(p_wseq, p_wlen)
        tmp2 = self.f_decoder(p_fseq, p_flen, t_hidden, w_hidden) # (B, maxlen, hidden)
        tmp3 = F.log_softmax(tmp2, dim=2)
        # add corresponding dim together to compute the total log likelihood
        B = p_fseq.shape[0]
        all_llk = []
        for i in range(B):
            d_llk = 0.
            for j in range(1,p_flen[i]):
                d_idx = p_fseq[i,j]
                d_llk += tmp3[i,j-1,d_idx]
            all_llk.append(d_llk.item())
        return all_llk

with open("./static/stop_model5_dataset.pkl","rb") as f:
    model4_dataset = pickle.load(f)

with open("./static/demo-ngram-size3.txt","r") as f:
    raw_ngram_3 = f.readlines()
with open("./static/demo-ngram-size4.txt","r") as f:
    raw_ngram_4 = f.readlines()
with open("./static/demo-ngram-size5.txt","r") as f:
    raw_ngram_5 = f.readlines()
mor_ngram = {
    3:[],
    4:[],
    5:[],
}
# should replace inner_join to join
# NOTICE:
# for testing, we only need <SOS> and len-1 tokens
# e.g., to test A,B,C, we need <SOS>,A,B to compute A,B,C
for d_raw in raw_ngram_3:
    tmp1 = d_raw.replace("inner_join","join")
    tmp2 = tmp1.strip().split()
    assert len(tmp2)==3, "tmp2 is: {}".format(tmp2)
    tmp3 = [model4_dataset["func_dict"][d_token] for d_token in tmp2]
    mor_ngram[3].append(tmp3)
for d_raw in raw_ngram_4:
    tmp1 = d_raw.replace("inner_join","join")
    tmp2 = tmp1.strip().split()
    assert len(tmp2)==4
    tmp3 = [model4_dataset["func_dict"][d_token] for d_token in tmp2]
    mor_ngram[4].append(tmp3)
for d_raw in raw_ngram_5:
    tmp1 = d_raw.replace("inner_join","join")
    tmp2 = tmp1.strip().split()
    assert len(tmp2)==5
    tmp3 = [model4_dataset["func_dict"][d_token] for d_token in tmp2]
    mor_ngram[5].append(tmp3)

def morpheus_get_hints(p_text):
    # functions from morpheus ngram that returns all possible function names
    return re.compile('\w+').findall(p_text)

def generate_one_ranking(arg_one_description, arg_one_size):
    descriptions = [arg_one_description]
    # formalize descriptions for every benchmark
    bmrk_descriptions = [] # equal to questions
    bmrk_hints = [] # equal to hints
    bmrk_titles = []
    for i in range(len(descriptions)):
        d_title = descriptions[i][0]
        d_question = descriptions[i][1]
        
        d_ques_str = " ".join( [p[1] if p[0]==0 else "" for p in d_question] ).lower()
        while d_ques_str.find("  ")>=0:
            d_ques_str = d_ques_str.replace("  "," ")
        nlp_ques = nlp(d_ques_str.lower())
        # t_ques = [d_token.text for d_token in nlp_ques]
        # ########## gen5 stop ########## #
        t_ques = [d_token.lemma_ for d_token in nlp_ques if (not d_token.is_stop) and (not d_token.is_punct) and (not d_token.is_space)]
        
        nlp_titl = nlp(d_title.lower())
        # t_titl = [d_token.text for d_token in nlp_titl]
        # ########## gen5 stop ########## #
        t_titl = [d_token.lemma_ for d_token in nlp_titl if (not d_token.is_stop) and (not d_token.is_punct) and (not d_token.is_space)]
        
        t_hint_list = [morpheus_get_hints(p[1]) if p[0]==0 else "" for p in d_question]
        
        u_ques = []
        for d_token in t_ques:
            if d_token in model4_dataset["word_dict"]:
                u_ques.append(model4_dataset["word_dict"][d_token])
            else:
                u_ques.append(model4_dataset["word_dict"]["<UKN>"])
                
        u_titl = []
        for d_token in t_titl:
            if d_token in model4_dataset["word_dict"]:
                u_titl.append(model4_dataset["word_dict"][d_token])
            else:
                u_titl.append(model4_dataset["word_dict"]["<UKN>"])
        
        u_hint = []
        u_hint.append(model4_dataset["hint_dict"]["<SEP>"]) # #### gen5 #### #
        for d_list in t_hint_list:
            need_sep = False
            for d_token in d_list:
                if d_token in model4_dataset["hint_dict"]:
                    u_hint.append(model4_dataset["hint_dict"][d_token])
                    need_sep = True
                # else do nothing
            if need_sep:
                u_hint.append(model4_dataset["hint_dict"]["<SEP>"]) # ####
                need_sep = False
        
        bmrk_descriptions.append(u_ques)
        bmrk_hints.append(u_hint)
        bmrk_titles.append(u_titl)

    # formalize functions/traversals for every benchmark
    # temp rule "sum"=="summarise"
    model4_dataset["func_dict"]["sum"]=model4_dataset["func_dict"]["summarise"]

    # construct data for benchmark ranking
    benchmark_dataset = {
        "word_list": model4_dataset["word_list"],
        "word_dict": model4_dataset["word_dict"],
        "hint_list": model4_dataset["hint_list"],
        "hint_dict": model4_dataset["hint_dict"],
        "func_list": model4_dataset["func_list"],
        "func_dict": model4_dataset["func_dict"],
    }
    bmrk_tokenized_data = []
    bmrk_tokenized_data_ind = []
    for i in range(len(bmrk_descriptions)):
        t_titl = bmrk_titles[i]
        t_ques = bmrk_descriptions[i]
        t_hint = bmrk_hints[i]
        
        for j in range(len(mor_ngram[arg_one_size])):
            # should add one <SOS> to all of them
            c_ansr = [model4_dataset["func_dict"]["<SOS>"]] + mor_ngram[arg_one_size][j]
            bmrk_tokenized_data.append(
                (t_titl, t_ques, t_hint, c_ansr)
            )
            bmrk_tokenized_data_ind.append(i)
    benchmark_dataset["tokenized_data"] = bmrk_tokenized_data
    benchmark_dataset["tokenized_data_ind"] = bmrk_tokenized_data_ind

    dt_bmrk_stack = StackDataset(p_dataset=benchmark_dataset, p_tmaxlen=20, p_wmaxlen=200, p_hmaxlen=40, p_fmaxlen=10)
    ld_bmrk_stack = DataLoader(dataset=dt_bmrk_stack, batch_size=256, shuffle=False)

    model = DoubleSeq2Seq(
        p_titl_vocab_size=len(dt_bmrk_stack.word_list),
        p_word_vocab_size=len(dt_bmrk_stack.word_list),
        p_func_vocab_size=len(dt_bmrk_stack.func_list),
    )
    if use_cuda:
        model.load_state_dict(torch.load("./static/models/r01_stop_model5_double_ep4.pt", map_location="cpu"))
        model = model.cuda()
        model.move_to_gpu()
    else:
        model.load_state_dict(torch.load("./static/models/r01_stop_model5_double_ep4.pt", map_location="cpu"))

    model.eval()
    all_scores = []
    n_total = len(dt_bmrk_stack)
    n_done = 0
    for batch_idx, (d_tseqs, d_wseqs, d_hseqs, d_fseqs, d_tlens, d_wlens, d_hlens, d_flens) in enumerate(ld_bmrk_stack):
        print("\r# Processing batch {}, {}/{} done".format(batch_idx,n_done,n_total),end="")
        td_tseqs = d_tseqs
        td_wseqs = d_wseqs
        td_hseqs = d_hseqs
        td_fseqs = d_fseqs
        
        td_tlens = d_tlens
        td_wlens = d_wlens
        td_hlens = d_hlens
        td_flens = d_flens
        if use_cuda:
            td_tseqs = td_tseqs.cuda()
            td_wseqs = td_wseqs.cuda()
            td_hseqs = td_hseqs.cuda()
            td_fseqs = td_fseqs.cuda()
        d_output = model.score(td_tseqs, td_wseqs, td_fseqs, td_tlens, td_wlens, td_flens)
        all_scores = all_scores+d_output
        n_done += d_wseqs.shape[0]
    print()
    print("# Done.")

    # #### gen5 modifications #### #
    # #### notice: we do not include d_titl in the pairs #### #
    expanded_pairs = {
        i:[] for i in range(len(bmrk_descriptions))
    }
    for i in range(len(all_scores)):
        d_titl = benchmark_dataset["tokenized_data"][i][0]
        d_ques = benchmark_dataset["tokenized_data"][i][1]
        d_hint = benchmark_dataset["tokenized_data"][i][2]
        d_ansr = benchmark_dataset["tokenized_data"][i][3][1:] # recover remove <SOS>
        d_ind = benchmark_dataset["tokenized_data_ind"][i]
        d_soc = all_scores[i]
        expanded_pairs[d_ind].append(
            (d_ques, d_ansr, d_soc, None)
        )

    sorted_expanded_pairs = {
        i:sorted(expanded_pairs[i], key=lambda x:x[2], reverse=True) for i in range(len(bmrk_descriptions))
    }

    # construct new ngram list
    new_ngram_list = []
    for i in range(len(bmrk_descriptions)):
        for j in range(len(sorted_expanded_pairs[i])):
            curr_ngram = [benchmark_dataset["func_list"][p] for p in sorted_expanded_pairs[i][j][1]]
            new_ngram_list.append(" ".join(curr_ngram).replace("join","inner_join"))

    return new_ngram_list


