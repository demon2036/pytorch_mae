import torch
import torch.nn as nn
import torch.nn.functional as F


class MoD(nn.Module):
    """
    Paper: https://arxiv.org/abs/2404.02258
    """

    def __init__(self, seq_len, capacity_factor, dim) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.capacity_factor = capacity_factor
        self.dim = dim

        # self.block=Bloxk()

        self.transformer_decoder_block = nn.Linear(self.dim, self.dim, bias=False)
        self.router = nn.Linear(self.dim, 1, bias=False)

    def forward(
            self, x,
    ):
        batch_size, seq_len, dim = x.shape

        top_k = int(seq_len * self.capacity_factor)

        # print(top_k, seq_len)

        router_logits = self.router(x)  # (x) batch,seq_len,dim -> r batch,seq_len,1

        #  𝑟𝑙> 𝑃𝛽 (R)  ... eqution 1

        class TopkWithGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, input, top_k):
                # 在 forward 方法中定义前向计算，input 是输入张量
                token_weights, token_index = torch.topk(input, top_k, dim=1, sorted=False)
                ctx.save_for_backward(token_weights, token_index)  # 保存中间结果，以备反向传播使用
                return token_index

            @staticmethod
            def backward(ctx, grad_output):
                # 在 backward 方法中定义反向传播，grad_output 是上游梯度
                token_weights, token_index = ctx.saved_tensors
                batch_size, seq_len, _ = grad_output.shape
                grad_input = torch.zeros(batch_size, seq_len, dtype=grad_output.dtype, device=grad_output.device)
                grad_input.scatter_add_(1, token_index, grad_output)  # 根据 token_index 将梯度累加到输入张量对应位置
                return grad_input, None  # 返回输入张量的梯度和 None（针对 top_k 参数的梯度为 None）

        token_index = TopkWithGradient.apply(router_logits, top_k)

        # token_weights, token_index = torch.topk(router_logits, top_k, dim=1, sorted=False)

        # print(router_logits, )
        # print(token_weights, token_weights.shape)
        # print(token_index, token_index.shape)

        token_index = token_index.expand(-1, -1, dim)

        # print(token_index.shape)

        select_x = torch.gather(x, 1, token_index)

        select_x = self.transformer_decoder_block(select_x)
        # select_x += 100
        # print('\n' * 5)
        # print(x)
        x = torch.scatter(x, 1, token_index, src=select_x)
        # print('\n' * 5)
        # print(x)

        return x


class MoeLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        experts = [nn.Linear(self.dim, self.dim, bias=False) for _ in range(8)]
        self.experts = nn.ModuleList(experts)
        self.router = nn.Linear(self.dim, 1, bias=False)

    def forward(self, inputs: torch.Tensor):
        gate_logits = self.router(inputs)
        weights, selected_experts = torch.topk(gate_logits, 1)

        print(weights.shape)
        print(selected_experts.shape)

        # while True:
        #     pass

        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            # results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
            #     inputs[batch_idx]
            # )

            results[batch_idx] += expert(
                inputs[batch_idx]
            )

        return results


if __name__ == "__main__":
    b, n, d = 2, 2, 2
    shape = (b, d)

    # mod_model = MoD(seq_len=b, capacity_factor=0.5, dim=d)
    mod_model = MoeLayer(d)

    x = torch.rand(shape)

    output = mod_model(x)

    target = torch.zeros_like(output, requires_grad=True)
    loss = F.l1_loss(target, output)

    loss.backward()

    print(loss)

    for p in mod_model.router.parameters():
        print(p.grad)
    # print(mod_model.router.parameters().gead)
