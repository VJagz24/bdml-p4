import torch
from torch.autograd import Function
import moe_dc_ext  # compiled cuda extension


class MoECombineFunction(Function):
    '''
    autograd func for MOE combine kernel
    foward: scatter outputs into original token orde
    backward:gathers gradients from token back into expert output slots
    and no computation for routing is needed
    '''
    @staticmethod
    def forward(ctx, expert_outputs, expert_indices, expert_counts, S):
        """
        expert_outputs: [E*capacity, C] concatenated output from all experts
        expert_indices: [E*capacity] maps each expert slot to orginal token index
        expert_counts:  [E] number of tokens routed to each expert
        S: int, total tokens total num of tokens
        returns combined output in original order
        """
        #first ensure we are on gpu then get dimension from input 
        assert expert_outputs.is_cuda, "expert_outputs must be CUDA"
        E_capacity, C = expert_outputs.shape
        E = expert_counts.shape[0]
        capacity = E_capacity // E

        #call the CUDA kernel
        y_flat = moe_dc_ext.combine(
            expert_outputs.contiguous(),
            expert_indices.contiguous(),
            expert_counts.contiguous(),
            int(S), int(C), int(E), int(capacity),
        )

        #save ttensors and metadata for backward
        ctx.save_for_backward(expert_indices, expert_counts)
        ctx.S = S
        ctx.C = C
        ctx.E = E
        ctx.capacity = capacity

        return y_flat

    @staticmethod
    def backward(ctx, grad_y_flat):
        """
        grad_y_flat: [S, C] gradient w.r.t combined output
        return: grad_expert_outputs, None, None, None
        """
        #get saved tensors 
        expert_indices, expert_counts = ctx.saved_tensors
        S = ctx.S
        C = ctx.C
        E = ctx.E
        capacity = ctx.capacity
        E_capacity = E * capacity

        device = grad_y_flat.device
        #allocate gradients tensors for expert outputs
        grad_expert_outputs = grad_y_flat.new_zeros((E_capacity, C))

        #check which slots were active
        slots = torch.arange(E_capacity, device=device, dtype=torch.int32)
        e = slots // capacity      
        s = slots % capacity       

        # check if slot is valid
        valid = s < expert_counts[e]

        # scatter gradients for valid slots
        valid_slots = valid.nonzero(as_tuple=True)[0]  # [N_valid]
        if valid_slots.numel() > 0:
            token_idx = expert_indices[valid_slots]    # [N_valid]
            grad_expert_outputs[valid_slots] = grad_y_flat[token_idx.long()]

        # no grads for indices, counts, S
        return grad_expert_outputs, None, None, None


def moe_combine_autograd(expert_outputs, expert_indices, expert_counts, S):
    return MoECombineFunction.apply(expert_outputs, expert_indices, expert_counts, S)