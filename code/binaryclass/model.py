import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################
# Mean + Fully connected NN
########################################################################################
class Mean(nn.Module):

    def __init__(self, out_dim):
        super(Mean, self).__init__()
        self.act_fn = nn.Tanh()
        self.linear = nn.Linear(out_dim, out_dim)

    def forward(self, feature):
        feature = self.linear(self.act_fn(feature))  # Tanh activation + Linear transform
        agg_vec_list = []
        for i in range(len(feature)):
            length = len(feature[i])
            #if torch.nonzero(att_mask[i] < 0, as_tuple=False).size(0) == 0:
            #    length = len(feature[i])
            #else:
            #    length = torch.nonzero(att_mask[i] < 0, as_tuple=False)[0] + 1
            agg_vec = torch.mean(feature[i][:length], dim=0)  # mean-pooling
            agg_vec_list.append(agg_vec)

        return torch.stack(agg_vec_list)  # (batch_size x feature_dim)

class LinearModel(nn.Module):
    def __init__(self, input_dim = 1024, output_class_num = 2):
        super(LinearModel, self).__init__()
        
        self.agg_method = Mean(input_dim)          # e.g., Mean()
        self.linear = nn.Linear(input_dim, output_class_num)   # classifier head
        #self.model = eval(config['module'])(config=Namespace(**config['hparams']))
        #self.head_mask = [None] * config['hparams']['num_hidden_layers']         

    def forward(self, features):
        #features = self.model(features, att_mask[:, None, None], head_mask=self.head_mask, output_all_encoded_layers=False)
        utterance_vector = self.agg_method(features)   # <- mean pooling
        predicted = self.linear(utterance_vector)                  # <- linear transformation
        return predicted


#代辦: 補回去att mask
