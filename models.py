import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3C_LSTM_GA(torch.nn.Module):

    def __init__(self, args):
        super(A3C_LSTM_GA, self).__init__()

        # Image Processing
        self.conv1 = nn.Conv2d(3, 128, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        # Instruction Processing
        self.gru_hidden_size = 256
        self.input_size = args.input_size
        self.embedding = nn.Embedding(self.input_size, 32)
        self.gru = nn.GRU(32, self.gru_hidden_size)

        # Gated-Attention layers
        self.text_linear = nn.Linear(self.gru_hidden_size, 64)#[self.gru_hidden_size=256]
        self.img_linear = nn.Linear(64, 64)
        self.attention_linear = nn.Linear(64,1)
        # self.addtable=nn.CAddTable()
        # self.dropout=nn.Dropout(0.5)
        # self.softmax=nn.SoftMax()
        # self.matrix_multiplication=nn.MM(false, false)


        # Time embedding layer, helps in stabilizing value prediction
        self.time_emb_dim = 32
        self.time_emb_layer = nn.Embedding(
                args.max_episode_length+1,
                self.time_emb_dim)

        # A3C-LSTM layers
        self.linear = nn.Linear(64 * 8 * 17, 256)
        self.lstm = nn.LSTMCell(256, 256)
        self.critic_linear = nn.Linear(256 + self.time_emb_dim, 1)
        self.actor_linear = nn.Linear(256 + self.time_emb_dim, 3)

        # Initializing weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()


    def forward(self, inputs):
        x, input_inst, (tx, hx, cx) = inputs

        # Get the image representation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x_image_rep = F.relu(self.conv3(x))#[1,64,8,17]

        # Get the instruction representation
        encoder_hidden = Variable(torch.zeros(1, 1, self.gru_hidden_size))
        for i in range(input_inst.data.size(1)):
            word_embedding = self.embedding(input_inst[0, i]).unsqueeze(0)
            _, encoder_hidden = self.gru(word_embedding, encoder_hidden)
        x_instr_rep = encoder_hidden.view(encoder_hidden.size(1), -1)

        # Get the attention vector from the instruction representation
        x_text_rep = F.sigmoid(self.text_linear(x_instr_rep)) #[1,64]
        # print('xatt',x_text_rep.size())
        # # Gated-Attention
        # x_attention1 = x_text_rep.unsqueeze(2).unsqueeze(3)#[1,64,1,1]
        # print('xatt1',x_attention1.size())
        # x_attention2 = x_attention1.expand(1, 64, 8, 17)#[1,64,8,17]
        # print('xatt2',x_attention2.size())
        # assert x_image_rep.size() == x_attention2.size()
        # print('ximg',x_image_rep.size())

        # ('text_emb_expand', (1L, 136L, 64L))
        # ('img_emb_dim_3', (1L, 136L, 64L))
        # ('text_emb_expand', (1L, 136L, 64L))
        # ('img_text_add', (1L, 136L, 64L))
        # ('h1_drop', (1L, 136L, 64L))
        # ('h1_drop_view', (136L, 64L))
        # ('h1_emb', (1L, 136L, 1L))
        # ('h1_emb_view', (1L, 136L))
        # ('img_text_add', (1L, 136L, 64L))
        # ('h1_drop', (1L, 136L, 64L))
        # ('h1_drop_view', (136L, 64L))
        # ('h1_emb', (1L, 136L, 1L))
        # ('h1_emb_view', (1L, 136L))
        # ('p1', (1L, 136L))
        # ('img_att1', (1L, 64L))
        # ('img_att_feat_1', (1L, 64L))
        # ('x_instruction_att', (1L, 64L))
        # ('x1', (1L, 136L, 64L))



        dim = 1#change from 0 to 1, due to acc=0, orneed to remove onely get warning
        # # Gated-Attention
        #image reshpae
        img_emb_dim_1=x_image_rep.view(-1,64)#[batch*img_seq_size,feat_dim]=[1*8*17,64]
        img_emb_dim_2 = self.img_linear(img_emb_dim_1)#[1*8*17,64]# local img_emb_dim_2 = nn.Linear(input_size, att_size, false)(nn.View(-1,input_size)(img_feat))
        img_emb_dim_3=img_emb_dim_2.view(1,8*17, 64)
        # print('img_emb_dim_3',img_emb_dim_3.size())
        #Text instruction reshpae
        text_emb_expand=x_text_rep.unsqueeze(1).repeat(1, 8*17, 1)#[1,8*17,64]
        # print('text_emb_expand',text_emb_expand.size())
        #Find Joint representation
        img_text_add=img_emb_dim_3+text_emb_expand#[1,8*17,64]# there is not caddtable in pytorch
        # print('img_text_add',img_text_add.size())
        h1=F.tanh(img_text_add)#[1,8*17,64]#[batch_size=1, att_size=64,input_size=64(if other than 64)] 
        h1_drop=F.dropout(h1)#[1,8*17,64]
        # print('h1_drop',h1_drop.size())
        #Attention embedding 
        h1_drop_view=h1_drop.view(-1,64)#[1*8*17,64]#[m=img_seq_size=8x17, att_size=64]
        # print('h1_drop_view',h1_drop_view.size())
        h1_emb=self.attention_linear(h1_drop)#[1,8*17,1]#local h1_emb = nn.Linear(att_size, 1)(nn.View(-1,att_size)(h2_drop)) -- [batch_size * m, 1]
        # print('h1_emb',h1_emb.size())#(1L, 136L, 1L)
        #Attention probability 
        h1_emb_view=h1_emb.view(-1,8*17)#(1L, 136L)#batch_size, m]
        # print('h1_emb_view',h1_emb_view.size())
        p1=F.softmax(h1_emb_view,dim)#batch_size, m]
        # print('p1',p1.size())#('p1', (1L, 136L))
        # p1_att=p1.view(1,1,8*17)#(a,b)-->(a,1,b) by x.view(a,1,b) which same as lua (nn.view(1,-1)):setNumInputDims(1)
        # print('p1_att',p1_att.size())#(1L, 1L, 136L)

        img_emb_dim_batch=img_emb_dim_3.view(8*17,64)
        # Weighted sum
        img_att1=torch.mm(p1,img_emb_dim_batch)  #[batch_size, 1, 64]
        # print('img_att1',img_att1.size())
        img_att_feat_1=img_att1.view(-1, 64)#[batch_size, 64]
        # print('img_att_feat_1',img_att_feat_1.size())
        x_instruction_att = x_text_rep+img_att_feat_1 
        # print('x_instruction_att',x_instruction_att.size())
        #[batch_size=200,m=8x17,attentionsize=64,input dim=256]

        x1=x_instruction_att.unsqueeze(1).repeat(1, 8*17, 1)#[1,8*17,64]
        # print('x1',x1.size())
        #x = x_image_rep*x_text_rep
        x = x1.view(x1.size(0), -1)

        # A3C-LSTM
        x = F.relu(self.linear(x))
        hx, cx = self.lstm(x, (hx, cx))
        time_emb = self.time_emb_layer(tx)
        x = torch.cat((hx, time_emb.view(-1, self.time_emb_dim)), 1)

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
