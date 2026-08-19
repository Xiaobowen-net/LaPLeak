import torch
"""
@acknowledgment:This code is based on the a publicly available code repository.<https://github.com/Koukyosyumei/Attack_SplitNN>
"""

class Client(torch.nn.Module):
    def __init__(self, client_model):
        super().__init__()
        self.client_model = client_model
        self.client_side_intermidiate = None
        self.grad_from_server = None

    def forward(self, inputs):
        self.client_side_intermidiate = self.client_model(inputs)
        # send intermidiate tensor to the server
        intermidiate_to_server = self.client_side_intermidiate.detach()\
            .requires_grad_()

        return intermidiate_to_server

    def client_backward(self, grad_from_server):
        self.grad_from_server = grad_from_server
        self.client_side_intermidiate.backward(grad_from_server)

    def train(self):
        self.client_model.train()

    def eval(self):
        self.client_model.eval()

class Server(torch.nn.Module):
    def __init__(self, server_model):
        super().__init__()
        self.server_model = server_model
        self.intermidiate_to_server = None
        self.intermidiate_to_top = None
        self.grad_to_client = None
        self.grad_from_top = None

    def forward(self, intermidiate_to_server):
        self.intermidiate_to_server = intermidiate_to_server
        self.server_side_intermidiate = self.server_model(intermidiate_to_server)

        # send intermidiate tensor to the server
        intermidiate_to_top = self.server_side_intermidiate.detach() \
            .requires_grad_()
        return intermidiate_to_top

    def server_backward(self, grad_from_top):
        self.grad_from_top = grad_from_top
        self.server_side_intermidiate.backward(grad_from_top)

        self.grad_to_client = self.intermidiate_to_server.grad.clone()
        return self.grad_to_client

    def train(self):
        self.server_model.train()

    def eval(self):
        self.server_model.eval()


class Top(torch.nn.Module):
    def __init__(self, top_model):
        super().__init__()
        self.top_model = top_model
        self.intermidiate_to_top = None
        self.grad_to_server = None

    def forward(self, intermidiate_to_top):
        self.intermidiate_to_top = intermidiate_to_top
        outputs = self.top_model(intermidiate_to_top)
        return outputs

    def top_backward(self):
        self.grad_to_server = self.intermidiate_to_top.grad.clone()
        return self.grad_to_server

    def train(self):
        self.top_model.train()

    def eval(self):
        self.top_model.eval()

class SplitNN(torch.nn.Module):
    def __init__(self, client, server, top, client_optimizer, server_optimizer, top_optimizer):
        super().__init__()
        self.client = client
        self.server = server
        self.top = top

        self.client_optimizer = client_optimizer
        self.server_optimizer = server_optimizer
        self.top_optimizer = top_optimizer

        self.intermidiate_to_server = None
        self.intermidiate_to_top =None
        self.intermidiate_grad = None

    def forward(self, inputs):

        self.intermidiate_to_server = self.client(inputs)
        self.intermidiate_to_top = self.server(self.intermidiate_to_server)
        outputs = self.top(self.intermidiate_to_top)
        return outputs

    def backward(self):

        grad_to_server = self.top.top_backward()

        grad_to_client = self.server.server_backward(grad_to_server)
        # execute client - back propagation
        self.client.client_backward(grad_to_client)

    def zero_grads(self):
        self.client_optimizer.zero_grad()
        self.server_optimizer.zero_grad()
        self.top_optimizer.zero_grad()

    def step(self):
        self.client_optimizer.step()
        self.server_optimizer.step()
        self.top_optimizer.step()

    def train(self):
        self.client.train()
        self.server.train()
        self.top.train()

    def eval(self):
        self.client.eval()
        self.server.eval()
        self.top.eval()

