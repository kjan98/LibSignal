import copy
from . import RLAgent
import random
import numpy as np
from collections import deque
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.registry import Registry
# import torch.optim as optim
# from torchsummary import summary

@Registry.register_model('model_frap')
class FRAP_DQNAgent(RLAgent):
    
    # def __init__(self, action_space, ob_generator, reward_generator, world, config, iid):
    #     super().__init__(action_space, ob_generator, reward_generator)
    def __init__(self, world, iid, prefix):
        super().__init__(world,iid)
        
        self.world = world
        # self.iid = iid
        self.ob_length = self.ob_generator.ob_length

        self.dic_agent_conf = Registry.mapping['model_mapping']['model_setting']
        self.dic_traffic_env_conf = convert_str2int(Registry.mapping['world_mapping']['traffic_setting'])
        self.buffer_size = Registry.mapping['task_mapping']['task_setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.dic_agent_conf.param["max_len"])
        # self.learning_start = Registry.mapping['task_mapping']['task_setting'].param['learning_start']
        # self.update_model_freq = Registry.mapping['task_mapping']['task_setting'].param["update_model_rate"]
        # self.update_target_model_freq = Registry.mapping['task_mapping']['task_setting'].param["update_target_rate"]
        self.gamma = self.dic_agent_conf.param["gamma"]
        self.epsilon = self.dic_agent_conf.param["epsilon"]
        self.epsilon_min = self.dic_agent_conf.param["epsilon_min"]
        self.epsilon_decay = self.dic_agent_conf.param["epsilon_decay"]
        self.learning_rate = self.dic_agent_conf.param["learning_rate"]
        self.batch_size = self.dic_agent_conf.param["batch_size"]

        self.num_phases = len(self.dic_traffic_env_conf.param["phase"])
        self.num_actions = len(self.dic_traffic_env_conf.param["phase"])

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

        self.action = 0
        self.last_action = 0
        self.replay_buffer = []
        self.if_test = 0

        self.optimizer = torch.optim.Adam(self.model.parameters(
        ), lr=0.005, eps=1e-08)
        self.loss_func = nn.MSELoss()

    def _build_model(self):
        model = FRAP(
            self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.ob_length, self.num_phases)
        return model

    def convert_state_to_input(self, s):
        inputs = {}
        if self.num_phases == 2:
            dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion_4_lane"]
        else:
            dic_phase_expansion = self.dic_traffic_env_conf.param["phase_expansion"]
        for feature in self.dic_traffic_env_conf.param["list_state_feature"]:
            if feature == "cur_phase":
                inputs[feature] = np.array([dic_phase_expansion[s[feature]+1]])
            else:
                inputs[feature] = np.array([s[feature]])
        return inputs

    def to_tensor(self, state):
        output = {}
        for i in state:
            output[i] = torch.from_numpy(state[i]).float()
        return output

    def get_action(self, ob):
        if not self.if_test and np.random.rand() <= self.epsilon:
            self.action = self.action_space.sample()
            return self.action
        state = {}
        state["cur_phase"] = self.world.id2intersection[self.id].current_phase
        self.last_action = state["cur_phase"]
        state["lane_num_vehicle"] = ob
        state_ = self.to_tensor(self.convert_state_to_input(state))
        q_values = self.model(state_)
        self.action = torch.argmax(q_values, dim=1).item()
        return self.action

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def remember(self, ob, action, reward, next_ob):
        last_state = {"cur_phase": self.last_action, "lane_num_vehicle": ob}
        state = {"cur_phase": action, "lane_num_vehicle": next_ob}
        self.replay_buffer.append((self.convert_state_to_input(
            last_state), action, reward, self.convert_state_to_input(state)))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        samples = random.sample(self.replay_buffer, self.batch_size)
        for input_list, action, reward, next_input in samples:
            out = self.target_model(self.to_tensor(next_input), train=False)
            target = reward + self.gamma * torch.max(out, dim=1)[0]
            target_f = self.model(self.to_tensor(input_list), train=False)
            target_f[0][action] = target
        loss = self.loss_func(self.model(
            self.to_tensor(input_list), train=True), target_f)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().numpy()

    def load_model(self, e):
        # name = "frap_torch_{}_{}.pt".format(self.id, e)
        # model_name = os.path.join(dir, name)
        model_name = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model', f'{e}.pt')
        self.model = FRAP(
            self.dic_agent_conf, self.dic_traffic_env_conf, self.num_actions, self.ob_length, self.num_phases)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, e):
        # name = "frap_torch_{}_{}.pt".format(self.id, e)
        # model_name = os.path.join(dir, name)
        path = os.path.join(Registry.mapping['logger_mapping']['output_path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}.pt')
        torch.save(self.model.state_dict(), model_name)


class FRAP(nn.Module):
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, num_actions, ob_length, num_phases):
        super(FRAP, self).__init__()
        self.dic_input_node = {}
        self.feature_shape = {}
        self.num_actions = num_actions
        self.ob_length = ob_length
        self.num_phases = num_phases
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_agent_conf = dic_agent_conf
        for feature_name in self.dic_traffic_env_conf.param["list_state_feature"]:
            if "phase" in feature_name:  # cur_phase
                _shape = (
                    self.dic_traffic_env_conf.param["dic_feature_dim"]["d_" + feature_name],)
            else:  # vehicle
                _shape = (self.ob_length,)

            self.feature_shape[feature_name] = _shape[0]


        self.embedding_1 = nn.Embedding(2, 4)
        self.dense_2 = nn.Linear(1, 4)
        self.lane_embedding = nn.Linear(8, 16)
        self.relation_embedd = nn.Embedding(2, 4)
        self.conv_feature = nn.Conv2d(
            in_channels=32, out_channels=self.dic_agent_conf.param["d_dense"], kernel_size=1)
        self.conv_relation = nn.Conv2d(
            in_channels=4, out_channels=self.dic_agent_conf.param["d_dense"], kernel_size=1)
        self.hidden_layer1 = nn.Conv2d(
            in_channels=20, out_channels=self.dic_agent_conf.param["d_dense"], kernel_size=1)
        self.hidden_layer2 = nn.Conv2d(
            in_channels=20, out_channels=1, kernel_size=1)

    def _forward(self, feature_list):
        for feature_name in feature_list:
            self.dic_input_node[feature_name] = feature_list[feature_name]
        p = F.sigmoid(self.embedding_1(
            self.dic_input_node["cur_phase"].long()))
        self.dic_input_node["lane_num_vehicle"] = self.remove_right_lane()
        dic_lane = {}
        for i, m in enumerate(self.dic_traffic_env_conf.param["list_lane_order"]):
            tmp = slice_tensor(
                x=self.dic_input_node["lane_num_vehicle"], index=i)
            tmp_vec = F.sigmoid(self.dense_2(tmp))
            tmp_phase = slice_tensor(x=p, index=i)
            dic_lane[m] = torch.cat([tmp_vec, tmp_phase])
        if self.num_actions == 8:
            list_phase_pressure = []
            for phase in self.dic_traffic_env_conf.param["phase"]:
                m1, m2 = phase.split("_")
                tmp1 = F.relu(self.lane_embedding(dic_lane[m1]))
                tmp2 = F.relu(self.lane_embedding(dic_lane[m2]))
                list_phase_pressure.append(tmp1.add(tmp2))

        elif self.num_actions == 4:
            list_phase_pressure = []
            for phase in self.dic_traffic_env_conf.param["phase"]:
                m1, m2 = phase.split("_")
                list_phase_pressure.append(torch.cat(
                    [dic_lane[m1], dic_lane[m2]], name=phase))

        constant = relation(
            x=self.dic_input_node["lane_num_vehicle"], dic_traffic_env_conf=self.dic_traffic_env_conf)
        relation_embedding = self.relation_embedd(constant)

        # rotate the phase pressure
        if self.dic_agent_conf.param["rotation"]:
            list_phase_pressure_recomb = []
            num_phase = self.num_phases
            for i in range(num_phase):
                for j in range(num_phase):
                    if i != j:
                        list_phase_pressure_recomb.append(
                            torch.cat([list_phase_pressure[i], list_phase_pressure[j]]))
            list_phase_pressure_recomb = torch.cat(list_phase_pressure_recomb)

            feature_map = torch.reshape(list_phase_pressure_recomb, shape=(
                -1, self.num_actions, self.num_actions-1, 32))
            feature_map = feature_map.permute(0, 3, 1, 2)
            lane_conv = F.relu(self.conv_feature(feature_map))
            relation_embedding = relation_embedding.permute(0, 3, 1, 2)
            if self.dic_agent_conf.param["merge"] == "multiply":
                relation_conv = self.conv_relation(relation_embedding)
                combine_feature = lane_conv*relation_conv
            elif self.dic_agent_conf.param["merge"] == "concat":
                relation_conv = self.conv_relation(relation_embedding)
                combine_feature = torch.cat(lane_conv, relation_conv)
            elif self.dic_agent_conf.param["merge"] == "weight":
                relation_conv = self.conv_relation(relation_embedding)
                tmp_wei = (lambda x: x.repeat(1, 1, 5))(relation_conv)
                combine_feature = lane_conv*tmp_wei

            hidden_layer = F.relu(self.hidden_layer1(combine_feature))
            before_merge = self.hidden_layer2(hidden_layer)
            before_merge = torch.reshape(before_merge, shape=(
                -1, self.num_actions, self.num_actions-1))
            q_values = (lambda x: torch.sum(x, dim=2))(before_merge)

        return q_values

    def forward(self, feature_list, train=True):
        if train:
            return self._forward(feature_list)
        else:
            with torch.no_grad():
                return self._forward(feature_list)
    
    def remove_right_lane(self):
        if self.dic_input_node["lane_num_vehicle"].size()[-1] == 12:
            N = self.dic_input_node["lane_num_vehicle"][0][1:3]
            E = self.dic_input_node["lane_num_vehicle"][0][4:6]
            S = self.dic_input_node["lane_num_vehicle"][0][7:9]
            W = self.dic_input_node["lane_num_vehicle"][0][10:12]
            lane_num_vehicle = torch.cat([E,S,W,N],dim=0)
            lane_num_vehicle = torch.unsqueeze(lane_num_vehicle, 0)
            return lane_num_vehicle
        else:
            return self.dic_input_node["lane_num_vehicle"]

def slice_tensor(x, index):
    x_shape = x.shape
    if len(x_shape) == 3:
        return x[:, index, :][0]
    elif len(x_shape) == 2:
        return x[:, index]


def relation(x, dic_traffic_env_conf):
    relations = []
    for p1 in dic_traffic_env_conf.param["phase"]:
        zeros = [0, 0, 0, 0, 0, 0, 0]
        count = 0
        for p2 in dic_traffic_env_conf.param["phase"]:
            if p1 == p2:
                continue
            m1 = p1.split("_")
            m2 = p2.split("_")
            if len(list(set(m1 + m2))) == 3:
                zeros[count] = 1
            count += 1
        relations.append(zeros)
    relations = np.array(relations).reshape(1, 8, 7)
    batch_size = x.shape[0]
    constant = torch.from_numpy(relations)
    constant = constant.repeat(batch_size, 1, 1)
    return constant

def convert_str2int(traffic_settings):
        """
        convert from string to number,including:
        traffic setting[phase_expansion]
        traffic setting[phase_expansion_4_lane]
        """
        dic={}
        for x in traffic_settings.param['phase_expansion']:
            dic[int(x)]=traffic_settings.param['phase_expansion'][x]
        traffic_settings.param['phase_expansion'].clear()    
        traffic_settings.param['phase_expansion']=copy.deepcopy(dic)
        
        dic.clear()
        
        for x in traffic_settings.param['phase_expansion_4_lane']:
            dic[int(x)]=traffic_settings.param['phase_expansion_4_lane'][x]
        traffic_settings.param['phase_expansion_4_lane'].clear()
        traffic_settings.param['phase_expansion_4_lane']=dic
        return traffic_settings