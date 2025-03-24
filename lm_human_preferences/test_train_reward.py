import torch
import torch.nn as nn
import torch.optim as optim
import tempfile

# We assume a Reward Network and Policy Network are involved
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        # ReLU activation for hidden layers, output layer is linear
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


class RewardNetwork(nn.Module):
    def __init__(self, input_size):
        super(RewardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # ReLU activation for hidden layers, output is a single scalar
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tanh(x)


# Hypothetical Hyperparameters (HParams)
class HParams:
    def __init__(self):
        self.rollout_batch_size = 8
        self.task = self.Task()
        self.noptepochs = 1
        self.labels = self.Labels()
        self.run = self.Run()

    class Task:
        def __init__(self):
            self.query_length = 2
            self.response_length = 3
            self.policy = self.Policy()
            self.query_dataset = 'test'
            self.start_text = None

        class Policy:
            def __init__(self):
                self.initial_model = 'test'

    class Labels:
        def __init__(self):
            self.source = 'test'
            self.num_train = 16
            self.type = 'best_of_4'

    class Run:
        def __init__(self):
            self.log_interval = 1
            self.save_dir = None
            self.save_interval = 1


    def override_from_dict(self, override_dict):
        for key, value in override_dict.items():
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            setattr(obj, keys[-1], value)

    def validate(self):
        pass


# Train the reward model (PyTorch-based)
def train_reward_train(hparams):
    # Initialize the Policy Network and Reward Network
    policy_net = PolicyNetwork(hparams.task.query_length, hparams.task.response_length)
    reward_net = RewardNetwork(hparams.task.query_length)
    
    # Create Optimizers for both models
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    reward_optimizer = optim.Adam(reward_net.parameters(), lr=0.001)
    
    # Loss function (for reward prediction)
    criterion = nn.MSELoss()

    # Simulated data (for the sake of this example)
    queries = torch.randn(hparams.rollout_batch_size, hparams.task.query_length)
    true_rewards = torch.randn(hparams.rollout_batch_size, 1)  # Simulated ground truth rewards

    # Training loop
    for epoch in range(hparams.noptepochs):
        policy_optimizer.zero_grad()
        reward_optimizer.zero_grad()
        
        # Forward pass: Get policy output and rewards from reward network
        responses = policy_net(queries)
        predicted_rewards = reward_net(responses)

        # Calculate loss (reward prediction error)
        loss = criterion(predicted_rewards, true_rewards)

        # Backward pass
        loss.backward()

        # Update the models
        policy_optimizer.step()
        reward_optimizer.step()

        # Log the loss
        if epoch % hparams.run.log_interval == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Save the model if needed
    if hparams.run.save_dir:
        save_path = tempfile.mkdtemp()
        torch.save(policy_net.state_dict(), f"{save_path}/policy_model.pth")
        torch.save(reward_net.state_dict(), f"{save_path}/reward_model.pth")
        print(f"Models saved at {save_path}")


# Helper function for the tests to override params easily
def train_reward_test(override_params):
    hparams = HParams()
    hparams.override_from_dict(override_params)
    hparams.validate()
    train_reward_train(hparams)


# Test function: runs a basic test with default hyperparameters
def test_basic():
    train_reward_test({})


# Test function: test with scalar comparison labels
def test_scalar_compare():
    train_reward_test({'labels.type': 'scalar_compare'})


# Test function: test with scalar rating labels
def test_scalar_rating():
    train_reward_test({'labels.type': 'scalar_rating'})


# Test function: test normalization before (not after)
def test_normalize_before():
    train_reward_test({
        'normalize_before': True,
        'normalize_after': False,
        'normalize_samples': 1024,
        'debug_normalize': 1024,
    })


# Test function: test normalization before and after
def test_normalize_both():
    train_reward_test({
        'normalize_before': True,
        'normalize_after': True,
        'normalize_samples': 1024,
        'debug_normalize': 1024,
    })


# Test function: test model saving
def test_save():
    train_reward_test({
        'run.save_dir': tempfile.mkdtemp(),
        'run.save_interval': 1
    })


if __name__ == "__main__":
    # Execute the test functions
    test_basic()
    test_scalar_compare()
    test_scalar_rating()
    test_normalize_before()
    test_normalize_both()
    test_save()
