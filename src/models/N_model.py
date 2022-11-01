from collections import defaultdict

from models.graph_learning_Attention.probsparseattention import GraphLearningProbSparseAttention
from models.Block import *
from models.layer.none_graph_learning_layer import None_Graph_Learning
from models.IC_PN_BEATS import attn_to_edge_index
from torch_geometric.utils import to_dense_adj
from utils.utils import build_batch_edge_index, build_batch_edge_weight


class N_model(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'
    N_HITS_BLOCK = 'n_hits'

    def __init__(self, config):
        super(N_model, self).__init__()

        self.config = config
        self.nodes_num = config.dataset.nodes_num
        self.num_feature = config.dataset.node_features
        self.n_head = config.graph_learning.n_head
        self.batch_size = config.train.batch_size

        if self.config.graph_learning.graph_learning:
            self.graph_learning_module = GraphLearningProbSparseAttention(self.config)
        else:
            self.graph_learning_module = None_Graph_Learning(self.config)

        self.attn_matrix = []

        self.activation = config.forecasting_module.activ

        self.stack_types = config.forecasting_module.stack_types
        self.num_blocks_per_stack = config.forecasting_module.num_blocks_per_stack
        self.share_weights_in_stack = config.forecasting_module.shapre_weights_in_stack

        self.inter_correlation_block_type = config.forecasting_module.inter_correlation_block_type
        self.forecast_length = config.forecasting_module.forecast_length
        self.backcast_length = self.forecast_length * 3
        self.n_theta_hidden = config.forecasting_module.n_theta_hidden
        self.thetas_dim = config.forecasting_module.thetas_dim
        self.n_layers = config.forecasting_module.inter_correlation_stack_length
        self.pooling_mode = config.forecasting_module.pooling_mode
        self.n_pool_kernel_size = config.forecasting_module.n_pool_kernel_size
        self.n_stride_size = config.forecasting_module.n_stride_size
        self.n_freq_downsample = config.forecasting_module.n_freq_downsample

        if (self.config.dataset.name == 'METR-LA') or (self.config.dataset.name == 'PEMS-BAY'):
            self.update_only_message = config.forecasting_module.update_only_message
        else:
            self.update_only_message = False

        self.stacks = []
        self.parameters = []

        for stack_id in range(len(self.stack_types)):
            self.stacks.append(self.create_stack(stack_id))

        self.parameters = nn.ParameterList(self.parameters)

    def create_stack(self, stack_id):
        stack_type = self.stack_types[stack_id]

        blocks = []
        for block_id in range(self.num_blocks_per_stack):
            block_init = N_model.select_block(stack_type)
            if self.share_weights_in_stack and block_id != 0:
                block = blocks[-1]
            else:
                if stack_type == N_model.N_HITS_BLOCK:
                    thetas_dim = [self.backcast_length,
                                  max(self.forecast_length // self.n_freq_downsample[stack_id], 1)]

                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden,
                                       thetas_dim=thetas_dim,
                                       backcast_length=self.backcast_length,
                                       forecast_length=self.forecast_length,
                                       inter_correlation_stack_length=self.n_layers,
                                       update_only_message=self.update_only_message,
                                       pooling_mode=self.pooling_mode,
                                       n_pool_kernel_size=self.n_pool_kernel_size[stack_id],
                                       n_stride_size=self.n_stride_size[stack_id])

                elif stack_type == N_model.TREND_BLOCK:
                    thetas_dim = [3, 3]

                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                                       backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                       inter_correlation_stack_length=self.n_layers,
                                       update_only_message=self.update_only_message)

                elif stack_type == N_model.SEASONALITY_BLOCK:
                    thetas_dim = [2 * int(self.backcast_length / 2 - 1) + 1,
                                  2 * int(self.forecast_length / 2 - 1) + 1]

                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden, thetas_dim=thetas_dim,
                                       backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                       inter_correlation_stack_length=self.n_layers,
                                       update_only_message=self.update_only_message)

                elif stack_type == N_model.GENERIC_BLOCK:
                    block = block_init(inter_correlation_block_type=self.inter_correlation_block_type,
                                       n_theta_hidden=self.n_theta_hidden, thetas_dim=self.thetas_dim,
                                       backcast_length=self.backcast_length, forecast_length=self.forecast_length,
                                       inter_correlation_stack_length=self.n_layers,
                                       update_only_message=self.update_only_message)

                self.parameters.extend(block.parameters())
            blocks.append(block)
        return blocks

    @staticmethod
    def select_block(block_type):
        if block_type == N_model.SEASONALITY_BLOCK:
            return Seasonlity_Block
        elif block_type == N_model.TREND_BLOCK:
            return Trend_Block
        elif block_type == N_model.GENERIC_BLOCK:
            return Generic_Block
        elif block_type == N_model.N_HITS_BLOCK:
            return N_HiTS_Block

        else:
            raise ValueError("Invalid block type")

    def forward(self, inputs, interpretability=False):
        inputs = inputs.squeeze()
        outputs = defaultdict(list)

        device = inputs.device

        forecast = torch.zeros(size=(inputs.size()[0], self.forecast_length)).to(device=device)
        backcast = torch.zeros(size=(inputs.size()[0], self.backcast_length)).to(device=device)

        _per_stack_backcast = []
        _per_stack_forecast = []

        _total_forecast_output = []
        _total_backcast_output = []

        _attention_matrix = []

        for stack_id in range(len(self.stacks)):
            stacks_forecast = torch.zeros(size=(inputs.size()[0], self.forecast_length)).to(device=device)
            stacks_backcast = torch.zeros(size=(inputs.size()[0], self.backcast_length)).to(device=device)

            if self.config.graph_learning.graph_learning:
                attn = self.graph_learning_module(
                    inputs.view(self.batch_size, self.nodes_num, self.backcast_length))
                batch_edge_index, batch_edge_weight = attn_to_edge_index(attn)
            else:
                edge_index, edge_attr = self.graph_learning_module()
                batch_edge_index = build_batch_edge_index(edge_index, num_graphs=self.batch_size,
                                                          num_nodes=self.nodes_num)
                attn = to_dense_adj(edge_index)[0]

                if edge_attr is None:
                    batch_edge_weight = None
                else:
                    batch_edge_weight = build_batch_edge_weight(edge_attr, num_graphs=self.batch_size)

            for block_id in range(len(self.stacks[stack_id])):
                b, f = self.stacks[stack_id][block_id](inputs, batch_edge_index, batch_edge_weight)

                inputs = inputs - b

                stacks_backcast = stacks_backcast + b
                stacks_forecast = stacks_forecast + f

            if interpretability:
                _per_stack_backcast.append(stacks_backcast.cpu().numpy())
                _per_stack_forecast.append(stacks_forecast.cpu().numpy())
                _attention_matrix.append(attn.cpu().detach().numpy())

            forecast = forecast + stacks_forecast
            backcast = backcast + stacks_backcast

        if interpretability:
            outputs['per_stack_backcast'] = np.stack(_per_stack_backcast, axis=0)
            outputs['per_stack_forecast'] = np.stack(_per_stack_forecast, axis=0)
            outputs['attention_matrix'] = np.stack(_attention_matrix, axis=0)

        return backcast, forecast, outputs


