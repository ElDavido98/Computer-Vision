from utils import *
from ResNet import ResNet
from EncDec import Encoder, PoolHiddenNet


ids = {
    "file_769": "1DqG01uXRIHWUnCxlbLgsiMQLy76eA7so",
    "file_770": "1vrc8-0U76FqpzXCo-8kpCdjOJbgsvseb",
    "file_771": "1POFZ2wQm0ouO4N96EY7ra5nQtkU5dtuN",
    "file_775": "1Q-aigGGqhsWBChiam3QGXWU8IpAW1KS5",
    "file_776": "1OdaV0RtN87UsqKP93M3JjRwD1VF9EgIW",
    "file_777": "1pY86QZ2oOtM-W9kB2J34ZDmAr5c_g_tt",
    "file_778": "1BpFnnt7TbFNITEjR1FUTIDe0ClidRtPi",
    "file_779": "19fgu7OApFdQ3vxuFJrqFRMJsBWKP1fLv",
    "file_780": "1Z1bl4aGfV00a3pMCuT_TW2UJd9HTi7dS",
    "file_781": "1QjWopaPpgjMWFtN5isx1sk1Ig78XzcDf",
    "file_782": "1s4JOujjUWI_LZC11IO8gHwXnf9CEYt8x",
    "file_783": "1DSigZpp9ErYUcC9EdT9MDP81_UsGhtsI",
    "file_784": "1f6Y7k-GHq2-ayPDYF-dzvX5Nq9e2erta",
    "file_785": "1Rm2zFV-OZ2jU48wg_W3uDKkitLxECh06"
}


def create_data(past_elements: int = 8, future_elements: int = 8, device=torch.device("cpu")):
    dataset = []

    ids_start = torch.zeros((1, 1, 1), device=device)
    scene_start = torch.zeros((1, 90, 160, 3), device=device)
    traffic_light_start = torch.zeros((1, 1), device=device)
    position_start = torch.zeros((1, 1, 2), device=device)
    states_start = torch.zeros((1, 1, 4), device=device)

    min_pos = None
    min_state = None
    max_pos = None
    max_state = None

    for i in range(769, 786):   # Dataset
        # Skip datasets without traffic light information
        if 774 >= i >= 772:
            continue
        print(i)

        data_i = download_data(name=f"file_{i}", file_id=ids[f"file_{i}"])
        length = len(data_i)

        for idx in range(past_elements - 1, length - future_elements):
            curr_agents = data_i[idx]["trajectories"]  # Agents in the frame
            num_agents = len(curr_agents)
            # Preallocate tensors for past, current, and future data
            past_and_curr_agent_ids = ids_start.repeat(past_elements, num_agents, 1)
            past_and_curr_scene = scene_start.repeat(past_elements, 1, 1, 1)
            past_and_curr_traffic_light = traffic_light_start.repeat(past_elements, 1)
            past_and_curr_positions = position_start.repeat(past_elements, num_agents, 1)
            past_and_curr_states = states_start.repeat(past_elements, num_agents, 1)
            future_agent_ids = ids_start.repeat(future_elements, num_agents, 1)
            # future_scene = scene_start.repeat(future_elements, 1, 1, 1)
            # future_traffic_light = traffic_light_start.repeat(future_elements, 1)
            future_positions = position_start.repeat(future_elements, num_agents, 1)
            # future_states = states_start.repeat(future_elements, num_agents, 1)

            t1, t2 = 1, 1
            for j in range(idx + future_elements, idx - past_elements, -1):
                # Scene information (1 BEV image)
                scene = torch.tensor(data_i[j]["scene_data"], dtype=torch.float32, device=device)
                # Traffic light sequence (same for every agent in the frame)
                traffic_light = torch.tensor(map_tl_to_code(data_i[j]["traffic_data"]), dtype=torch.float32, device=device)

                agent_ids = torch.zeros((num_agents,), dtype=torch.long, device=device)
                position = torch.zeros((num_agents, 2), dtype=torch.float32, device=device)
                states = torch.zeros((num_agents, 4), dtype=torch.float32, device=device)

                for a in range(num_agents):
                    for b in range(len(data_i[j]["trajectories"])):
                        if len(data_i[j]["trajectories"]) > a and data_i[j]["trajectories"][b][0] == curr_agents[a][0]:
                            agent_ids[a] = curr_agents[a][0]
                            # Positions (X, Y) of the considered agent
                            position[a, :] = torch.tensor([data_i[j]["trajectories"][b][1],
                                                           data_i[j]["trajectories"][b][2]
                                                           ],
                                                          device=device)
                            min_pos = min_values(min_pos, position[a, :])
                            max_pos = max_values(max_pos, position[a, :])
                            # State (SPEED, TAN_ACC, LAT_ACC, ANG) of the considered agent
                            states[a, :] = torch.tensor([data_i[j]["trajectories"][b][3],
                                                         data_i[j]["trajectories"][b][4],
                                                         data_i[j]["trajectories"][b][5],
                                                         data_i[j]["trajectories"][b][6]
                                                         ],
                                                        device=device)
                            min_state = min_values(min_state, states[a, :])
                            max_state = max_values(max_state, states[a, :])
                            break

                if j <= idx:
                    past_and_curr_agent_ids[idx - j - t1, :, :] = agent_ids.unsqueeze(-1)
                    past_and_curr_scene[idx - j - t1, :, :, :] = scene.unsqueeze(0)
                    past_and_curr_traffic_light[idx - j - t1, :] = traffic_light.unsqueeze(0)
                    past_and_curr_positions[idx - j - t1, :, :] = position
                    past_and_curr_states[idx - j - t1, :, :] = states
                    t1 += 2
                else:
                    future_agent_ids[idx + future_elements - j - t2, :, :] = agent_ids.unsqueeze(-1)
                    # future_scene[idx + future_elements - j - t2, :, :, :] = scene.unsqueeze(0)
                    # future_traffic_light[idx + future_elements - j - t2, :] = traffic_light.unsqueeze(0)
                    future_positions[idx + future_elements - j - t2, :, :] = position
                    # future_states[idx + future_elements - j - t2, :, :] = states
                    t2 += 2

            dataset.append({"past_and_curr": [past_and_curr_agent_ids, past_and_curr_scene,
                                              past_and_curr_traffic_light, past_and_curr_positions, past_and_curr_states],
                            # "future": [future_agent_ids, future_scene, future_traffic_light, future_positions,
                            # future_states]})
                            "future": [future_agent_ids, future_positions]})

    return dataset, [min_pos, max_pos], [min_state, max_state]


def train_val_test_split(
        dataset: list,
        train_perc: float = 0.60,
        validation_perc: float = 0.23,
        test_perc: float = 0.17
):
    random.shuffle(dataset)

    length = len(dataset)
    train_cutoff = int(length * train_perc)
    validation_cutoff = int(length * validation_perc) + train_cutoff
    test_cutoff = int(length * test_perc) + validation_cutoff

    train_dataset = dataset[:train_cutoff]
    validation_dataset = dataset[train_cutoff:validation_cutoff]
    test_dataset = dataset[validation_cutoff:test_cutoff]

    return train_dataset, validation_dataset, test_dataset


def download_data(name: str = "my_dataset", file_id: str = None):
    current_directory = os.path.dirname(os.path.abspath(__file__))

    gdown.download(id=file_id, output=os.path.join(current_directory, f"{name}.json"))

    with open(f"{name}.json", "r", encoding="utf8") as file:
        data = json.load(file)

    return data


def save_dataset(
        data_set: list,
        name: str,
):
    torch.save(data_set, name)


def upload_dataset(name: str):
    data = torch.load(name)
    return data


class Processor(nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            in_height: int = 360,
            in_width: int = 640,
            out_channels: int = 64,
            hidden_channels: int = 64,
            num_blocks: int = 18,
            kernel_size: int = 7,
            stride: int = 1,
            padding: int = 3,
            embedding_dim: int = 64,
            hidden_dim: int = 64,
            num_layers: int = 1,
            batch_size: int = 8,
            dropout: float = 0.0,
            bottleneck_dim: int = 1024,
            concat_dim: int = 320,
            flag: list = None,
            device=torch.device("cpu")
    ):
        super(Processor, self).__init__()
        if flag is None:
            flag = [0, 0, 0]
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.flag = flag
        self.device = device

        self.traj_pool_MLP = make_multilayer(
            in_out_dims=[hidden_dim + bottleneck_dim, 2 * (hidden_dim + bottleneck_dim), hidden_dim + bottleneck_dim],
            dropout=dropout, 
            batch_norm_features=batch_size,
            device=device
        )

        if self.flag[0] == 0:
            out_height = int((in_height + 2 * padding - kernel_size) / stride + 1)
            out_width = int((in_width + 2 * padding - kernel_size) / stride + 1)
            self.scene_context_encoder = ResNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                kernel_size=7,
                stride=1,
                padding=3,
                device=device
            )
            self.scene_projector = nn.Linear(out_channels * out_height * out_width, self.hidden_dim)
        if self.flag[1] == 0:
            self.traffic_light_encoder = Encoder(
                in_features=1,
                embedding_dim=embedding_dim,
                batch_size=1,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device
            )
        self.trajectory_encoder = Encoder(
            in_features=2,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            device=device
        )
        if self.flag[2] == 0:
            self.state_encoder = Encoder(
                in_features=4,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                device=device
                                        )
        self.pooling_module = PoolHiddenNet(
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
            device=device
        )
        self.transformation = make_multilayer(
            in_out_dims=[concat_dim, 2 * concat_dim, concat_dim],
            dropout=dropout,
            batch_norm_features=batch_size,
            batch_norm=True,
            device=device
        )

    def __call__(
            self,
            x: torch.Tensor,
            min_max_pos: torch.Tensor,
            min_max_state: torch.Tensor,
    ):
        y = self.forward(x, min_max_pos, min_max_state)
        return y

    def integrate(
            self,
            scene_feature: torch.Tensor,
            signal_feature: torch.Tensor,
            state_feature: torch.Tensor,
            trajectory_feature: torch.Tensor,
            trajectory_features_pool: torch.Tensor
    ):
        concatenated_tensor = None

        trajectory_feature = self.traj_pool_MLP(torch.cat([trajectory_feature, trajectory_features_pool.view([self.num_layers, self.batch_size, self.bottleneck_dim])], dim=-1))

        if self.flag[0] == 0:
            _, out_channels, height, width = scene_feature.size()
            scene_projector = nn.Linear(out_channels * height * width, self.hidden_dim, device=self.device)
            scene_feature_flat = scene_feature.reshape(self.batch_size, out_channels * height * width)
            scene_feature_projected = scene_projector(scene_feature_flat).unsqueeze(0)
            if self.flag[1:3] == [0, 0]:
                concatenated_tensor = torch.cat([trajectory_feature, state_feature, signal_feature,
                                                 scene_feature_projected], dim=-1)
            if self.flag[1:3] == [1, 0]:
                concatenated_tensor = torch.cat([trajectory_feature, state_feature, scene_feature_projected], dim=-1)
            if self.flag[1:3] == [0, 1]:
                concatenated_tensor = torch.cat([trajectory_feature, signal_feature, scene_feature_projected], dim=-1)
            if self.flag[1:3] == [1, 1]:
                concatenated_tensor = torch.cat([trajectory_feature, scene_feature_projected], dim=-1)
        else:
            if self.flag[1:3] == [0, 0]:
                concatenated_tensor = torch.cat([trajectory_feature, state_feature, signal_feature], dim=-1)
            if self.flag[1:3] == [1, 0]:
                concatenated_tensor = torch.cat([trajectory_feature, state_feature], dim=-1)
            if self.flag[1:3] == [0, 1]:
                concatenated_tensor = torch.cat([trajectory_feature, signal_feature], dim=-1)
            if self.flag[1:3] == [1, 1]:
                concatenated_tensor = trajectory_feature

        transformed_features = self.transformation(concatenated_tensor)

        return transformed_features.permute(1, 0, 2)

    def forward(
            self,
            x: torch.Tensor,
            min_max_pos: torch.Tensor,
            min_max_state: torch.Tensor,
    ):
        _, image, traffic_light, trajectory, state = x

        scene_features, traffic_light_features, trajectory_features, state_features = None, None, None, None

        if self.flag[0] == 0:
            scene_features = self.scene_context_encoder(min_max_normalize_images(image.to(self.device)))

        if self.flag[1] == 0:
            traffic_light_features = self.traffic_light_encoder(traffic_light.to(self.device))
            traffic_light_features = traffic_light_features.repeat(1, 8, 1)

        if self.flag[2] == 0:
            state_features = self.state_encoder(
                min_max_normalize(
                    state.to(self.device),
                    min_max_state[0].to(self.device),
                    min_max_state[1].to(self.device)
                )
            )

        normalized_trajectory = min_max_normalize(
            trajectory.to(self.device),
            min_max_pos[0].to(self.device),
            min_max_pos[1].to(self.device)
        )
        trajectory_features = self.trajectory_encoder(normalized_trajectory.to(self.device))
        trajectory_features_pool = self.pooling_module(trajectory_features, normalized_trajectory[-1, :, :])

        data = self.integrate(
            scene_features,
            traffic_light_features,
            state_features,
            trajectory_features,
            trajectory_features_pool
        )

        return data
