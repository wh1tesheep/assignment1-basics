from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    """
    给定 Linear 层的权重，计算批量输入的线性变换。

    参数：
        in_dim (int)：输入维度的大小。
        out_dim (int)：输出维度的大小。
        weights (Float[Tensor, "d_out d_in"])：要使用的线性层权重。
        in_features (Float[Tensor, "... d_in"])：要应用该函数的输入 Tensor。

    返回：
        Float[Tensor, "... d_out"]：线性模块变换后的输出。
    """

    raise NotImplementedError


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    给定 Embedding 层的权重，获取一批 token ID 对应的嵌入表示。

    参数：
        vocab_size (int)：词表中的嵌入数量。
        d_model (int)：嵌入维度的大小。
        weights (Float[Tensor, "vocab_size d_model"])：从中查找嵌入向量的权重。
        token_ids (Int[Tensor, "..."])：要从 Embedding 层中查找的一组 token ID。

    返回：
        Float[Tensor, "... d_model"]：Embedding 层返回的一批嵌入表示。
    """

    raise NotImplementedError


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 SwiGLU 网络的权重，返回你的实现使用这些权重得到的输出。

    参数：
        d_model (int)：前馈网络输入和输出的维度。
        d_ff (int)：SwiGLU 内部上投影的维度。
        w1_weight (Float[Tensor, "d_ff d_model"])：保存的 W1 权重。
        w2_weight (Float[Tensor, "d_model d_ff"])：保存的 W2 权重。
        w3_weight (Float[Tensor, "d_ff d_model"])：保存的 W3 权重。
        in_features (Float[Tensor, "... d_model"])：前馈层的输入嵌入。

    返回：
        Float[Tensor, "... d_model"]：与输入嵌入 shape 相同的输出嵌入。
    """
    # 示例：
    # 如果状态字典中的键名匹配，可以使用 `load_state_dict()`
    # swiglu.load_state_dict(weights)
    # 也可以手动赋值权重
    # swiglu.w1.weight.data = w1_weight
    # swiglu.w2.weight.data = w2_weight
    # swiglu.w3.weight.data = w3_weight
    raise NotImplementedError


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    给定键（K）、查询（Q）和值（V）Tensor，返回你的缩放点积注意力实现的输出。

    参数：
        Q (Float[Tensor, " ... queries d_k"])：查询 Tensor。
        K (Float[Tensor, " ... keys d_k"])：键 Tensor。
        V (Float[Tensor, " ... keys d_v"])：值 Tensor。
        mask (Bool[Tensor, " ... queries keys"] | None)：掩码 Tensor。
    返回：
        Float[Tensor, " ... queries d_v"]：SDPA 的输出。
    """
    raise NotImplementedError


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定朴素、无批次的多头注意力实现中的键、查询和值投影权重，返回优化后的批处理实现的输出。
    该实现应通过一次矩阵乘法处理所有注意力头的键、查询和值投影。
    此函数不应使用 RoPE。
    参见 Vaswani 等人（2017）的第 3.2.2 节。

    参数：
        d_model (int)：前馈网络输入和输出的维度。
        num_heads (int)：多头注意力使用的注意力头数量。
        max_seq_len (int)：如果实现会预先缓存，则为要缓存的最大序列长度。
        q_proj_weight (Float[Tensor, "d_model d_model"])：Q 投影的权重。
        k_proj_weight (Float[Tensor, "d_model d_model"])：K 投影的权重。
        v_proj_weight (Float[Tensor, "d_model d_model"])：V 投影的权重。
        o_proj_weight (Float[Tensor, "d_model d_model"])：输出投影的权重。
        in_features (Float[Tensor, "... sequence_length d_model"])：要交给你的实现处理的 Tensor。

    返回：
        Float[Tensor, " ... sequence_length d_model"]：使用给定 QKV 投影权重和输入特征运行优化后的
        批处理多头注意力实现所得到的输出 Tensor。
    """
    raise NotImplementedError


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    """
    给定朴素、无批次的多头注意力实现中的键、查询和值投影权重，返回优化后的批处理实现的输出。
    该实现应通过一次矩阵乘法处理所有注意力头的键、查询和值投影。
    此版本的 MHA 应包含 RoPE。
    在这种情况下，RoPE 嵌入维度必须等于单个注意力头的嵌入维度（d_model // num_heads）。
    参见 Vaswani 等人（2017）的第 3.2.2 节。

    参数：
        d_model (int)：前馈网络输入和输出的维度。
        num_heads (int)：多头注意力使用的注意力头数量。
        max_seq_len (int)：如果实现会预先缓存，则为要缓存的最大序列长度。
        theta (float)：RoPE 参数。
        q_proj_weight (Float[Tensor, "d_model d_model"])：Q 投影的权重。
        k_proj_weight (Float[Tensor, "d_model d_model"])：K 投影的权重。
        v_proj_weight (Float[Tensor, "d_model d_model"])：V 投影的权重。
        o_proj_weight (Float[Tensor, "d_model d_model"])：输出投影的权重。
        in_features (Float[Tensor, "... sequence_length d_model"])：要交给你的实现处理的 Tensor。
        token_positions (Int[Tensor, " ... sequence_length"] | None)：包含 token 位置的可选 Tensor。

    返回：
        Float[Tensor, " ... sequence_length d_model"]：使用给定 QKV 投影权重和输入特征运行优化后的
        批处理多头注意力实现所得到的输出 Tensor。
    """
    raise NotImplementedError


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    """
    对给定的输入 Tensor 应用 RoPE。

    参数：
        d_k (int)：查询或键 Tensor 的嵌入维度大小。
        theta (float)：RoPE 参数。
        max_seq_len (int)：如果实现会预先缓存，则为要缓存的最大序列长度。
        in_query_or_key (Float[Tensor, "... sequence_length d_k"])：要应用 RoPE 的输入 Tensor。
        token_positions (Int[Tensor, "... sequence_length"])：包含 token 位置、shape 为
            (batch_size, sequence_length) 的 Tensor。
    返回：
        Float[Tensor, " ... sequence_length d_k"]：应用 RoPE 后的输入 Tensor。
    """
    raise NotImplementedError


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    """
    给定 pre-norm Transformer block 的权重和输入特征，返回在这些输入特征上运行该 block 的输出。

    此函数应使用 RoPE。
    根据你的实现方式，你可能只需将相关参数传给 TransformerBlock 构造函数，
    也可能需要自行初始化 RoPE 类并将其传入。

    参数：
        d_model (int)：Transformer block 输入的维度。
        num_heads (int)：多头注意力使用的注意力头数量；`d_model` 必须能被 `num_heads` 整除。
        d_ff (int)：前馈网络内部层的维度。
        max_seq_len (int)：如果实现会预先缓存，则为要缓存的最大序列长度。
        theta (float)：RoPE 参数。
        weights (dict[str, Tensor]):
            参考实现的状态字典。
            该字典包含以下键：
            - `attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                Shape 为 (d_model, d_model)。
                各行按照 shape 为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                Shape 为 (d_model, d_model)。
                各行按照 shape 为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                Shape 为 (d_model, d_model)。
                各行按照 shape 为 (num_heads, d_v) 的矩阵排列，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `attn.output_proj.weight`
                多头自注意力输出投影的权重。
                Shape 为 (d_model, d_model)。
            - `ln1.weight`
                Transformer block 中第一个 RMSNorm 所应用的仿射变换权重。
                Shape 为 (d_model,)。
            - `ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                Shape 为 (d_model, d_ff)。
            - `ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `ln2.weight`
                Transformer block 中第二个 RMSNorm 所应用的仿射变换权重。
                Shape 为 (d_model,)。
        in_features (Float[Tensor, "batch sequence_length d_model"]):
            要交给你的实现处理的 Tensor。

    返回：
        Float[Tensor, "batch sequence_length d_model"]：使用 RoPE 在输入特征上运行
        Transformer block 后得到的输出 Tensor。
    """
    raise NotImplementedError


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    """给定 Transformer 语言模型的权重和输入索引，返回对输入索引执行前向传播的输出。

    此函数应使用 RoPE。

    参数：
        vocab_size (int)：待预测输出词表中的不同元素数量。
        context_length (int)：一次最多处理的 token 数量。
        d_model (int)：模型嵌入和各子层输出的维度。
        num_layers (int)：使用的 Transformer 层数。
        num_heads (int)：多头注意力使用的注意力头数量；`d_model` 必须能被 `num_heads` 整除。
        d_ff (int)：前馈网络内部层的维度（第 3.3 节）。
        rope_theta (float)：RoPE 的 $\\Theta$ 参数。
        weights (dict[str, Tensor]):
            参考实现的状态字典。{num_layers} 表示 `0` 到 `num_layers - 1` 之间的整数（即层索引）。
            该字典包含以下键：
            - `token_embeddings.weight`
                Token 嵌入矩阵。Shape 为 (vocab_size, d_model)。
            - `layers.{num_layers}.attn.q_proj.weight`
                所有 `num_heads` 个注意力头的查询投影。
                Shape 为 (num_heads * (d_model / num_heads), d_model)。
                各行按照 shape 为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.k_proj.weight`
                所有 `num_heads` 个注意力头的键投影。
                Shape 为 (num_heads * (d_model / num_heads), d_model)。
                各行按照 shape 为 (num_heads, d_k) 的矩阵排列，
                因此 `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.v_proj.weight`
                所有 `num_heads` 个注意力头的值投影。
                Shape 为 (num_heads * (d_model / num_heads), d_model)。
                各行按照 shape 为 (num_heads, d_v) 的矩阵排列，
                因此 `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`。
            - `layers.{num_layers}.attn.output_proj.weight`
                多头自注意力输出投影的权重。
                Shape 为 ((d_model / num_heads) * num_heads, d_model)。
            - `layers.{num_layers}.ln1.weight`
                Transformer block 中第一个 RMSNorm 所应用的仿射变换权重。
                Shape 为 (d_model,)。
            - `layers.{num_layers}.ffn.w1.weight`
                FFN 中第一个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `layers.{num_layers}.ffn.w2.weight`
                FFN 中第二个线性变换的权重。
                Shape 为 (d_model, d_ff)。
            - `layers.{num_layers}.ffn.w3.weight`
                FFN 中第三个线性变换的权重。
                Shape 为 (d_ff, d_model)。
            - `layers.{num_layers}.ln2.weight`
                Transformer block 中第二个 RMSNorm 所应用的仿射变换权重。
                Shape 为 (d_model,)。
            - `ln_final.weight`
                应用于最终 Transformer block 输出的 RMSNorm 仿射变换权重。
                Shape 为 (d_model, )。
            - `lm_head.weight`
                语言模型输出嵌入的权重。
                Shape 为 (vocab_size, d_model)。
        in_indices (Int[Tensor, "batch_size sequence_length"])：用于运行语言模型的输入索引 Tensor。
            Shape 为 (batch_size, sequence_length)，其中 `sequence_length` 不超过 `context_length`。

    返回：
        Float[Tensor, "batch_size sequence_length vocab_size"]：包含每个 token 的下一词预测结果的 Tensor，
        这些预测值尚未归一化。
    """
    raise NotImplementedError


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    """给定 RMSNorm 仿射变换的权重，返回对输入特征运行 RMSNorm 后的输出。

    参数：
        d_model (int)：RMSNorm 输入的维度。
        eps (float)：为保证数值稳定性而加到分母上的数值。
        weights (Float[Tensor, "d_model"])：RMSNorm 权重。
        in_features (Float[Tensor, "... d_model"])：要应用 RMSNorm 的输入特征，
            可以具有任意前导维度。

    返回：
        Float[Tensor,"... d_model"]：对 `in_features` 运行 RMSNorm 后得到的、
        与 `in_features` shape 相同的 Tensor。
    """
    raise NotImplementedError


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """给定输入 Tensor，返回对每个元素应用 SiLU 后的输出。

    参数：
        in_features(Float[Tensor, "..."])：要应用 SiLU 的输入特征，shape 可以任意。

    返回：
        Float[Tensor,"..."]：对每个元素应用 SiLU 后得到的、与 `in_features` shape 相同的 Tensor。
    """
    raise NotImplementedError


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    给定数据集（由整数组成的一维 NumPy 数组）、期望的 batch size 和上下文长度，
    从数据集中采样语言模型输入序列及其对应标签。

    参数：
        dataset (np.array)：包含数据集整数 token ID 的一维 NumPy 数组。
        batch_size (int)：期望采样的 batch size。
        context_length (int)：每个采样样本期望的上下文长度。
        device (str)：PyTorch 设备字符串（例如 'cpu' 或 'cuda:0'），表示采样得到的
            输入序列和标签应放置在哪个设备上。

    返回：
        一对 shape 为 (batch_size, context_length) 的 torch.LongTensor。
        第一个元素是采样的输入序列，第二个元素是对应的语言模型标签。
    """
    raise NotImplementedError


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    给定输入 Tensor，返回沿输入的指定 `dim` 应用 softmax 后的输出。

    参数：
        in_features (Float[Tensor, "..."])：要应用 softmax 的输入特征，shape 可以任意。
        dim (int)：在 `in_features` 的哪个维度上应用 softmax。

    返回：
        Float[Tensor, "..."]：沿指定 `dim` 进行 softmax 归一化后得到的、
        与 `in_features` shape 相同的 Tensor。
    """
    raise NotImplementedError


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """给定输入和目标 Tensor，计算所有样本的平均交叉熵损失。

    参数：
        inputs (Float[Tensor, "batch_size vocab_size"])：inputs[i][j] 是第 i 个样本
            属于第 j 类的未归一化 logit。
        targets (Int[Tensor, "batch_size"])：shape 为 (batch_size,) 的 Tensor，
            其中包含正确类别的索引；每个值必须介于 0 和 `num_classes - 1` 之间。

    返回：
        Float[Tensor, ""]：所有样本的平均交叉熵损失。
    """
    raise NotImplementedError


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """给定一组参数，对它们的联合梯度进行裁剪，使其 L2 范数不超过 max_l2_norm。

    参数：
        parameters (Iterable[torch.nn.Parameter])：可训练参数的集合。
        max_l2_norm (float)：表示最大 L2 范数的正数。

    应就地修改参数的梯度（parameter.grad）。
    """
    raise NotImplementedError


def get_adamw_cls() -> Any:
    """
    返回一个实现 AdamW 的 torch.optim.Optimizer。
    """
    raise NotImplementedError


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    给定带线性预热的余弦学习率衰减调度参数以及迭代次数，
    返回指定调度在该次迭代时的学习率。

    参数：
        it (int)：要获取学习率的迭代次数。
        max_learning_rate (float)：alpha_max，带预热的余弦学习率调度中的最大学习率。
        min_learning_rate (float)：alpha_min，带预热的余弦学习率调度中的最小或最终学习率。
        warmup_iters (int)：T_w，线性预热学习率所需的迭代次数。
        cosine_cycle_iters (int)：T_c，余弦退火的迭代次数。

    返回：
        指定调度在给定迭代次数时的学习率。
    """
    raise NotImplementedError


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    给定模型、优化器和迭代次数，将它们序列化到磁盘。

    参数：
        model (torch.nn.Module)：要序列化其状态的模型。
        optimizer (torch.optim.Optimizer)：要序列化其状态的优化器。
        iteration (int)：要序列化的数值，表示已经完成的训练迭代次数。
        out (str | os.PathLike | BinaryIO | IO[bytes])：用于保存模型、优化器和迭代次数的
            路径或文件状对象。
    """
    raise NotImplementedError


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    给定序列化检查点（路径或文件状对象），将其中保存的状态恢复到给定模型和优化器，
    并返回此前序列化到检查点中的迭代次数。

    参数：
        src (str | os.PathLike | BinaryIO | IO[bytes])：序列化检查点的路径或文件状对象。
        model (torch.nn.Module)：要恢复状态的模型。
        optimizer (torch.optim.Optimizer)：要恢复状态的优化器。
    返回：
        int：此前序列化的迭代次数。
    """
    raise NotImplementedError


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """给定词表、合并列表和特殊 token 列表，返回使用这些内容的 BPE tokenizer。

    参数：
        vocab (dict[int, bytes])：Tokenizer 词表，即从 int（词表中的 token ID）
            到 bytes（token 字节）的映射。
        merges (list[tuple[bytes, bytes]])：BPE 合并。列表中的每一项都是一个 bytes 二元组
            (<token1>, <token2>)，表示将 <token1> 与 <token2> 合并。
            各项按照合并的创建顺序排列。
        special_tokens (list[str] | None)：Tokenizer 使用的字符串特殊 token 列表。
            这些字符串永远不会被拆成多个 token，并始终作为单个 token 保留。

    返回：
        使用所提供词表、合并列表和特殊 token 的 BPE tokenizer。
    """
    raise NotImplementedError


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """给定输入语料库的路径，训练 BPE tokenizer 并输出其词表和合并列表。

    参数：
        input_path (str | os.PathLike)：BPE tokenizer 训练数据的路径。
        vocab_size (int)：Tokenizer 词表的元素总数，其中包括特殊 token。
        special_tokens (list[str])：要加入 tokenizer 词表的字符串特殊 token 列表。
            这些字符串永远不会被拆成多个 token，并始终作为单个 token 保留。
            如果这些特殊 token 出现在 `input_path` 中，则把它们当作其他普通字符串处理。

    返回：
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                训练得到的 tokenizer 词表，即从 int（词表中的 token ID）
                到 bytes（token 字节）的映射。
            merges:
                BPE 合并。列表中的每一项都是一个 bytes 二元组 (<token1>, <token2>)，
                表示将 <token1> 与 <token2> 合并。
                各项按照合并的创建顺序排列。
    """
    raise NotImplementedError
