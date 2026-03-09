# gpu上
# pip3 install flash-linear-attention==0.4.1
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_h
import torch
#core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
#     query,
#     key,
#     value,
#     g=g,
#     beta=beta,
#     **chunk_kwargs,
# )

#self.chunk_gated_delta_rule info 
# (torch.Size([1, 12291, 32, 128]), 
# torch.Size([1, 12291, 32, 128]), 
# torch.Size([1, 12291, 32, 128]), 
# torch.Size([1, 12291, 32]), 
# torch.Size([1, 12291, 32]), 
# {'initial_state': None, 
# 'output_final_state': False, 'use_qk_l2norm_in_kernel': True, 
# 'cu_seqlens': tensor([    0,  4097,  8194, 12291], dtype=torch.int32)})

def test_gdn():
    query = torch.rand([1, 200, 32, 128]).cuda()*0.1
    key = torch.rand([1, 200, 32, 128]).cuda() *0.1
    value = torch.rand([1, 200, 32, 128]).cuda() *0.1
    g = torch.rand([1, 200, 32]).cuda()*0.1
    beta = torch.rand([1, 200, 32]).cuda()*0.1
    cu_seqlens = torch.Tensor([    0,  37,  101, 200]).to(torch.int32).cuda()
    chunk_kwargs = {
        "cu_seqlens": cu_seqlens,
        "output_final_state": False,
        'initial_state': None, 
        "use_qk_l2norm_in_kernel": True,
    }
    core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        **chunk_kwargs,
    )
    print(f"core_attn_out: {torch.max(core_attn_out), torch.min(core_attn_out)}, {last_recurrent_state}")


"""
与 vLLM chunk_gated_delta_rule_fwd_h 完全等价的 NumPy 实现。
参考: https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fla/ops/chunk_delta_h.py
"""
from typing import Optional, Tuple

import numpy as np


def _prepare_chunk_offsets(cu_seqlens: np.ndarray, chunk_size: int) -> np.ndarray:
    """与 vLLM prepare_chunk_offsets 等价：每个序列的 chunk 起始索引。"""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    num_chunks = (lens + chunk_size - 1) // chunk_size
    return np.concatenate([[0], np.cumsum(num_chunks)]).astype(cu_seqlens.dtype)


def chunk_gated_delta_rule_fwd_h_numpy(
    k: np.ndarray,
    w: np.ndarray,
    u: np.ndarray,
    g: Optional[np.ndarray] = None,
    initial_state: Optional[np.ndarray] = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Chunk-wise Gated Delta Rule 前向计算（与 vLLM 完全等价的 NumPy 实现）。

    输入:
        k: (B, T, Hg, K) 或 varlen 时 (1, total_T, Hg, K)
        w: (B, T, H, K)
        u: (B, T, H, V)  即原接口中的 v
        g: (B, T, H) 或 (1, total_T, H)，可选门控
        initial_state: (N, H, K, V) 可选初始状态
        output_final_state: 是否输出最终状态
        chunk_size: 块大小 BT，默认 64
        save_new_value: 是否保存 v_new（delta）
        cu_seqlens: 变长序列的累积长度 [0, l1, l1+l2, ...]，若为 None 则等长

    输出:
        h: (B, NT, H, K, V) 每个 chunk 开始时的状态
        v_new: (B, T, H, V) 或 None，delta 值 v - w@h
        final_state: (N, H, K, V) 或 None，最终状态
    """
    B, T, Hg, K = k.shape
    H = u.shape[2]
    V = u.shape[3]
    BT = chunk_size

    assert K <= 256, "K must be <= 256"
    assert w.shape == (B, T, H, K) and u.shape == (B, T, H, V)

    if cu_seqlens is None:
        N = B
        NT = (T + BT - 1) // BT
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        NT = int(_prepare_chunk_offsets(cu_seqlens, BT)[-1])
        chunk_offsets = _prepare_chunk_offsets(cu_seqlens, BT)

    dtype = k.dtype
    # 内部用 float32 累积状态（与 kernel 一致）
    h = np.zeros((B, NT, H, K, V), dtype=np.float32)
    final_state = np.zeros((N, H, K, V), dtype=np.float32) if output_final_state else None
    v_new = np.empty_like(u) if save_new_value else None

    for i_n in range(N):
        if cu_seqlens is None:
            bos = i_n * T
            eos = i_n * T + T
            seq_T = T
            boh = i_n * NT
            seq_NT = NT
        else:
            bos = int(cu_seqlens[i_n])
            eos = int(cu_seqlens[i_n + 1])
            seq_T = eos - bos
            boh = int(chunk_offsets[i_n])
            seq_NT = int(chunk_offsets[i_n + 1] - chunk_offsets[i_n])

        for i_h in range(H):
            # Hg: k 按 head 分组，head i_h 对应 k 的组索引
            hg = i_h // (H // Hg) if H >= Hg else 0

            # 当前 batch/head 的 k, w, u 视图 [seq_T, ...]
            k_nh = k[0, bos:eos, hg, :] if cu_seqlens is not None else k[i_n, :, hg, :]  # (seq_T, K)
            w_nh = w[0, bos:eos, i_h, :] if cu_seqlens is not None else w[i_n, :, i_h, :]  # (seq_T, K)
            u_nh = u[0, bos:eos, i_h, :] if cu_seqlens is not None else u[i_n, :, i_h, :]  # (seq_T, V)

            # 初始状态
            if initial_state is not None:
                state = initial_state[i_n, i_h, :, :].astype(np.float32).copy()  # (K, V)
            else:
                state = np.zeros((K, V), dtype=np.float32)

            for i_t in range(seq_NT):
                t_start = i_t * BT
                t_end = min(t_start + BT, seq_T)
                cur_BT = t_end - t_start

                # 1) 写出当前 chunk 开始时的状态 h（batch_idx, chunk_idx, head, K, V）
                batch_idx = 0 if cu_seqlens is not None else i_n
                chunk_idx = (boh + i_t) if cu_seqlens is not None else i_t
                h[batch_idx, chunk_idx, i_h, :, :] = state

                # 2) v_new = u - w @ h  (delta rule)
                w_chunk = w_nh[t_start:t_end, :]   # (cur_BT, K)
                u_chunk = u_nh[t_start:t_end, :]   # (cur_BT, V)
                v_new_chunk = u_chunk.astype(np.float32) - (w_chunk.astype(np.float32) @ state)

                if save_new_value:
                    if cu_seqlens is not None:
                        v_new[0, bos + t_start:bos + t_end, i_h, :] = v_new_chunk.astype(dtype)
                    else:
                        v_new[i_n, t_start:t_end, i_h, :] = v_new_chunk.astype(dtype)

                # 3) 可选门控：v_new *= exp(g_last - g_t)，h *= exp(g_last)
                if g is not None:
                    g_nh = g[0, bos:bos + seq_T, i_h] if cu_seqlens is not None else g[i_n, :, i_h]
                    last_idx = t_end - 1
                    g_last = g_nh[last_idx]
                    g_chunk = g_nh[t_start:t_end]
                    m_t = (np.arange(cur_BT) + t_start) < seq_T  # 与 kernel 中 (i_t*BT + arange(BT)) < T 等价
                    v_new_chunk = v_new_chunk * np.where(m_t, np.exp(g_last - g_chunk), 0)[:, np.newaxis]
                    g_last_exp = np.exp(g_last)
                    state = state * g_last_exp

                # 4) 状态更新: h += k^T @ v_new
                k_chunk = k_nh[t_start:t_end, :].T  # (K, cur_BT)
                state = state + (k_chunk.astype(np.float32) @ v_new_chunk.astype(k_chunk.dtype))

            if output_final_state:
                final_state[i_n, i_h, :, :] = state

    # 将 h 转为与 k 相同 dtype（kernel 中 store 时转换）
    h = h.astype(dtype)
    return h, v_new, final_state





def test_gdn_rule_fwd_h_compare():
    # chunk_gated_delta_rule_fwd_h input shape info: 
    # {'k': torch.Size([1, 9049, 32, 128]),
    #  'w': torch.Size([1, 9049, 32, 128]), 
    #  'u': torch.Size([1, 9049, 32, 128]), 
    #  'g': torch.Size([1, 9049, 32]), 
    #  'initial_state': None, 
    #  'cu_seqlens': tensor([   0, 4097, 8194, 9049], device='cuda:0', dtype=torch.int32)}
    # chunk_gated_delta_rule_fwd_h out info: 
    # {'h': torch.Size([1, 144, 32, 128, 128]), 
    # 'v_new': torch.Size([1, 9049, 32, 128]), 
    # 'final_state': None}
    k = torch.rand([1, 12384, 32, 128]).cuda().float()*0.01
    w = torch.rand([1, 12384, 32, 128]).cuda().float()*0.01
    u = torch.rand([1, 12384, 32, 128]).cuda().float()*0.01
    g = torch.rand([1, 12384, 32]).cuda().float()*0.01


    cu_seqlens = torch.Tensor([    0,  2099,  5078, 12384]).to(torch.int32).cuda()
    initial_state = None
    output_final_state = False
    h_cuda, v_new_cuda, final_state_cuda = chunk_gated_delta_rule_fwd_h(
        k=k.clone(),
        w=w.clone(),
        u=u.clone(),
        g=g.clone(),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens.clone(),
    )

    h_np, v_new_np, final_state_np = chunk_gated_delta_rule_fwd_h_numpy(
        k=k.cpu().numpy(),
        w=w.cpu().numpy(),
        u=u.cpu().numpy(),
        g=g.cpu().numpy(),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens.cpu().numpy(),
    )
    print(f"cuda out: {torch.max(h_cuda), torch.min(h_cuda), torch.max(v_new_cuda), torch.min(v_new_cuda)}")
    print(f"np out: {np.max(h_np), np.min(h_np), np.max(v_new_np), np.min(v_new_np)}")
    assert np.allclose(h_cuda.cpu().numpy(), h_np, atol=1e-5)
    assert np.allclose(v_new_cuda.cpu().numpy(), v_new_np, atol=1e-5)
    assert final_state_cuda is None and final_state_np is None

    # def chunk_gated_delta_rule_fwd_h_numpy(
    #     k: np.ndarray,
    #     w: np.ndarray,
    #     u: np.ndarray,
    #     g: Optional[np.ndarray] = None,
    #     initial_state: Optional[np.ndarray] = None,
    #     output_final_state: bool = False,
    #     chunk_size: int = 64,
    #     save_new_value: bool = True,
    #     cu_seqlens: Optional[np.ndarray] = None,)


# test_gdn()
test_gdn_rule_fwd_h_compare()