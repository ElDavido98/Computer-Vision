from utils import *
from metrics import *
from GAN import GAN


class Forecasting(nn.Module):
    def __init__(
            self,
            batch_size: int = 1,
            num_agents: int = 8,
            concat_dim: int = 320,
            flag: list = None,
            device=torch.device("cpu")
    ):
        super(Forecasting, self).__init__()

        if flag is None:
            flag = [0, 0, 0]

        self.batch_size = batch_size
        self.num_agents = num_agents
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.device = device

        self.GAN = GAN(
            in_channels=3,
            num_blocks=1,
            batch_size=self.num_agents,
            concat_dim=concat_dim,
            flag=flag,
            device=device
        )

    def train(
            self,
            train_set: list = None,
            validation_set: list = None,
            epochs: int = 50,
            name: str = 'model.pt'
    ):
        train_set, self.GAN.min_max_pos, self.GAN.min_max_state = train_set
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        if validation_set is not None:
            validation_set, self.GAN.min_max_pos, self.GAN.min_max_state = validation_set
            self.validation_loader = torch.utils.data.DataLoader(
                validation_set,
                batch_size=self.batch_size,
                shuffle=True
            )
        print("Start Training")
        for epoch in range(epochs):
            printProgressAction('Epoch ', epoch)
            self.GAN.pre_train_step()
            # Train
            print("   Train")
            for num, (train_batch) in enumerate(self.train_loader):
                printProgressAction('    Batch', num)
                past_data, target_trajectory = agents_selection(train_batch, self.num_agents, device=self.device)
                if len(past_data) == 0:
                    continue
                self.GAN.train_step(past_data, target_trajectory.to(self.device), num)
                if (num + 1) % 500 == 0:
                    self.save(name=name)
                # Clear unused memory
                torch.cuda.empty_cache()
            self.save(name=name)

            # Validation
            if epoch % 5 == 0 and validation_set is not None:
                print("\n   Validation")
                for num, (validation_batch) in enumerate(self.validation_loader):
                    printProgressAction('    Batch', num)
                    past_data, target_trajectory = agents_selection(validation_batch, self.num_agents, device=self.device)
                    if len(past_data) == 0:
                        continue
                    D_gen, D_targ = self.GAN.validation_step(past_data, target_trajectory.to(self.device))
                    self.GAN.post_validation_step(D_gen, D_targ)
                    # Clear unused memory
                    torch.cuda.empty_cache()
                # EarlyStopping
                self.GAN.stopping()
                if self.GAN.done:
                    print(f"Stopped prematurely at iteration {epoch} due to EarlyStopping")
                    break
        print("\nEnd Training")

    def evaluation(
            self,
            test_set: list,
    ):
        test_set, self.GAN.min_max_pos, self.GAN.min_max_state = test_set
        print("Start Evaluation")
        self.GAN.pre_test_step(len(test_set))
        self.test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        for num, (test_batch) in enumerate(self.test_loader):
            printProgressAction('    Batch', num)
            past_data, target_trajectory = agents_selection(test_batch, self.num_agents, device=self.device)
            if len(past_data) == 0:
                continue
            self.GAN.test_step(num, past_data)
            # Clear unused memory
            torch.cuda.empty_cache()

        # Final Evaluation Using Metrics
        ADE = average_displacement_error(self.GAN.eval_pred, self.GAN.eval_targ)
        FDE = final_displacement_error(self.GAN.eval_pred, self.GAN.eval_targ)
        AVG = average([ADE, FDE])
        print("\n        Average Displacement Error (ADE) : ", ADE)
        print("        Final Displacement Error (FDE) : ", FDE)
        print("        Average (AVG) : ", AVG)
        print("End Evaluation")

    def save(self, name: str = 'model.pt'):
        torch.save(self.state_dict(), name)

    def load(self, name: str = 'model.pt'):
        self.load_state_dict(torch.load(name, map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
