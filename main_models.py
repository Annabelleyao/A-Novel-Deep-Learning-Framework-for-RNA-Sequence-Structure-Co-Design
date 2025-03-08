import torch.nn as nn
import torch.nn.functional as F
import gvp
from data_processing import *

class Net(nn.Module):
    def __init__(self, in_dims, out_dims):
        """
        Define the layers of the convolutional neural network.

        Parameters:
            in_channels: int
                The number of channels in the input image. For MNIST, this is 1 (grayscale images).
            num_classes: int
                The number of classes we want to predict, in our case 10 (digits 0 to 9).
        """
        super(Net, self).__init__()

        # First convolutional layer:
        out_Dims1 = in_dims[0]*2, in_dims[1]*2
        self.conv1 = gvp.GVPConv(in_dims=in_dims, out_dims=out_Dims1, edge_dims=in_dims, activations=(
            F.relu, torch.sigmoid), vector_gate=True).double()
        # Second convolutional layer:
        # self.pool = gvp.GVPConv(in)
        # # Third convolutional layer:
        out_Dims2 = out_Dims1[0]*2, out_Dims1[1]*2
        # 现扩大再变小so there can be more characteristics created for model to learn from
        self.conv2 = gvp.GVPConv(
            in_dims=out_Dims1, out_dims=out_Dims2, edge_dims=in_dims).double()
        out_Dims3 = out_Dims2[0]*2, out_Dims2[1]*2
        self.conv3 = gvp.GVPConv(
            in_dims=out_Dims2, out_dims=out_Dims3, edge_dims=in_dims).double()
        self.conv4 = gvp.GVPConv(
            in_dims=out_Dims3, out_dims=out_dims, edge_dims=in_dims).double()

    def forward(self, x, edge_index, edges):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: torch.Tensor
                The input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        # x = self.conv1(x,edge_index,edges)  # Apply first convolution and ReLU activation
        # if edges.shape[1] != 19:  # Ensure edge_attr matches required dims
        #     padding = torch.zeros(edges.shape[0], 19 - edges.shape[1]).to(edges.device)
        #     edges = torch.cat((edges, padding), dim=1)
        x = self.conv1(x, edge_index, edges)
        x1 = (x[0].double(), x[1].double())
        x = self.conv2(x1, edge_index, edges)
        x1 = (x[0].double(), x[1].double())
        x = self.conv3(x1, edge_index, edges)
        x1 = (x[0].double(), x[1].double())
        x = self.conv4(x1, edge_index, edges)
        # x = self.conv2(x, edge_index, edges)
        # x = self.pool(x)           # Apply max pooling
        # S=x[0].double()
        # V=x[1].double()
        # x = (S,V)
        # x = (self.conv2(x,edge_index,edges))  # Apply second convolution and ReLU activation
        # x = self.pool(x)           # Apply max pooling
        # x = x.reshape(x.shape[0], -1)  # Flatten the tensor
        # x = self.fc1(x)            # Apply fully connected layer
        return x
    
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import gvp
import numpy as np
from data_processing import torsion, calculate_vector, get_rotation_frames, rotate

# Encoder_Network 定义（保留原有逻辑，不输出中间 out）
class Encoder_Network():
    def __init__(self, args): 
        self.allcord = {}
        self.in_dims = args.in_dims
        self.out_dims = args.out_dims
        self.GVP = self.build_GVP(self.in_dims, self.out_dims)

    def extract_coord(self, path):
        with open(path) as file:
            lines = file.readlines()
        atom_lines = [line for line in lines if line.startswith('ATOM')]
        length_rna = int(atom_lines[-1].split()[5])
        for i in range(length_rna):
            self.allcord[i] = dict()
        for line in atom_lines:
            parts = line.split()
            key = int(parts[5]) - 1
            self.allcord[key][parts[2]] = np.array([float(parts[6]), float(parts[7]), float(parts[8])])
    
    def build_GVP(self, in_dims, out_dims):
        return Net(in_dims, out_dims)
    
    def forward(self, path):
        self.extract_coord(path)
        pdb_torsion = dict()
        pdb_vector = dict()
        for i in range(len(self.allcord)):  # 每个核苷酸
            pdb_torsion[i] = torsion(i, self.allcord)
            pdb_vector[i] = calculate_vector(i, self.allcord)
        
        # 构建距离矩阵与邻居索引
        num_nodes = len(self.allcord.keys())
        distance_matrix = np.zeros((num_nodes, num_nodes))
        for i in self.allcord.keys():
            for j in self.allcord.keys():
                coord_i = np.array(self.allcord[i]["C4'"])
                coord_j = np.array(self.allcord[j]["C4'"])
                distance_matrix[i, j] = np.linalg.norm(coord_i - coord_j)
        nearest_neighbors = {}
        k = 5
        for i in self.allcord.keys():
            nearest_indices = np.argsort(distance_matrix[i])[:k+1]
            nearest_neighbors[i] = [j for j in nearest_indices if j != i][:k]
        
        # 构建节点特征：S 为标量特征，V 为向量特征
        S = torch.tensor([list(pdb_torsion[i].values()) for i in range(num_nodes)], dtype=torch.double)
        V = torch.tensor([pdb_vector[i] for i in range(num_nodes)], dtype=torch.double).unsqueeze(1)
        
        # 构建边的索引
        in_index = []
        out_index = []
        for i in nearest_neighbors.keys():
            for j in nearest_neighbors[i]:
                in_index += [i, j]
                out_index += [j, i]
        edge_index = torch.tensor([in_index, out_index], dtype=torch.int64)
        
        # 两层 GVP 卷积
        scalars_in, vectors_in = 8, 1
        scalars_out, vectors_out = 3, 5
        in_dims = (scalars_in, vectors_in)
        out_dims = (scalars_out, vectors_out)
        # 第一层随机边特征
        edges = gvp.randn(n=int(10*num_nodes), dims=in_dims)
        conv = gvp.GVPConv(in_dims, out_dims, edge_dims=in_dims).double()
        out1 = conv((S.double(), V.double()), edge_index, edges)
        
        # 第二层卷积
        scalars_in2, vectors_in2 = 3, 5
        scalars_out2, vectors_out2 = 5, 7
        in_dims2 = (scalars_in2, vectors_in2)
        out_dims2 = (scalars_out2, vectors_out2)
        S1 = out1[0].double()
        V1 = out1[1].double()
        nodes = (S1, V1)
        edges2 = gvp.randn(n=int(10*num_nodes), dims=in_dims2)
        conv2 = gvp.GVPConv(in_dims2, out_dims2, edge_dims=in_dims2,
                              activations=(F.relu, torch.sigmoid), vector_gate=True).double()
        out2 = conv2(nodes, edge_index, edges2)
        
        # 使用内部的 GVP 模型进一步卷积
        edges3 = gvp.randn(n=int(10*num_nodes), dims=self.in_dims)
        out3 = self.GVP(nodes, edge_index, edges3)
        
        # 对卷积后的向量进行 flatten
        x_flattened = out3[1].view(num_nodes, -1).float()
        
        # 利用原子坐标构造旋转矩阵
        coord = torch.zeros([num_nodes, 3, 3], dtype=torch.float64)
        for i in self.allcord:
            val = torch.stack([
                torch.tensor(self.allcord[i]['N1']),
                torch.tensor(self.allcord[i]["C1'"]),
                torch.tensor(self.allcord[i]["C4'"])
            ], dim=0)
            coord[i] = val
        coord = torch.unsqueeze(coord, 0)
        R = get_rotation_frames(coord)
        
        # 合并特征
        gvp_out_features = torch.cat([
            out3[0].float(),
            rotate(out3[1], R.transpose(-2, -1)).flatten(-2, -1).squeeze().float(),
            x_flattened
        ], dim=-1)
        # 通过一层全连接映射到最终特征维度（47维），方便解码器后续处理
        linear = nn.Linear(gvp_out_features.shape[-1], 47).float()
        final_features = linear(gvp_out_features)
        return final_features  # shape: (seq_len, 47)

# Transformer 解码器（保持原有 RNAGeneratorTransformer 逻辑）
class RNAGeneratorTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, seq_len_dim, num_heads=8, num_layers=6):
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        super(RNAGeneratorTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.seq_len_fc = nn.Linear(seq_len_dim, embedding_dim)
        self.position_encoding = nn.Embedding(1000, embedding_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x, seq_len_features):
        token_embeddings = self.embedding(x)  # (batch, seq_len, embedding_dim)
        seq_len_embeddings = self.seq_len_fc(seq_len_features).unsqueeze(1)  # (batch, 1, embedding_dim)
        token_embeddings = token_embeddings + seq_len_embeddings.expand(-1, token_embeddings.size(1), -1)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        position_embeddings = self.position_encoding(positions)
        token_embeddings = token_embeddings + position_embeddings
        output = self.transformer_decoder(token_embeddings, token_embeddings)
        output = self.fc_out(output)
        return output

# 定义一个新的联合网络，将编码器和解码器融合在一起
class CombinedNetwork(nn.Module):
    def __init__(self, encoder_args, vocab=None, embedding_dim=128, hidden_dim=128,
                 seq_len_dim=47, num_heads=8, num_layers=6):
        super(CombinedNetwork, self).__init__()
        self.encoder = Encoder_Network(encoder_args)
        # 如果没有提供词典，采用默认设置
        if vocab is None:
            self.vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '<START>': 4, '<END>': 5}
        else:
            self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        vocab_size = len(self.vocab)
        self.decoder = RNAGeneratorTransformer(vocab_size, embedding_dim, hidden_dim, seq_len_dim,
                                                 num_heads=num_heads, num_layers=num_layers)

    def forward(self, pdb_path):
        # 编码器：从 PDB 文件中提取特征，输出形状为 (seq_len, 47)
        encoder_features = self.encoder.forward(pdb_path)
        seq_len = encoder_features.shape[0]
        # 计算全局特征作为解码器输入：例如取均值，得到 (47,) 的向量
        seq_len_feature = encoder_features.mean(dim=0)
        # 解码器生成 RNA 序列（采用贪心生成）
        self.decoder.eval()  # 在推断时设为 eval 模式
        generated_sequence = [self.vocab['<START>']]
        input_tensor = torch.tensor(generated_sequence, dtype=torch.long).unsqueeze(0).to(encoder_features.device)
        # 生成长度为 seq_len 的序列（可以根据需要调整生成策略）
        for _ in range(seq_len + 1 - len(generated_sequence)):
            output = self.decoder(input_tensor, seq_len_feature.unsqueeze(0))
            logits = output[:, -1, :]
            # 屏蔽 <START> 与 <END>，使模型不会提前结束生成
            for token in [self.vocab['<START>'], self.vocab['<END>']]:
                logits[:, token] = -float('inf')
            next_token = torch.argmax(logits, dim=1).item()
            generated_sequence.append(next_token)
            input_tensor = torch.tensor(generated_sequence, dtype=torch.long).unsqueeze(0).to(encoder_features.device)
        # 转换成 RNA 字符串
        rna_sequence = ''.join(self.inverse_vocab[tok] for tok in generated_sequence if tok in self.inverse_vocab)
        return rna_sequence[1:]

# 参数类示例
class gvp_args():
    def __init__(self):
        self.scalars_in = 3
        self.vectors_in = 5
        self.scalars_out = 5
        self.vectors_out = 7
        self.in_dims = (self.scalars_in, self.vectors_in)
        self.out_dims = (self.scalars_out, self.vectors_out)
