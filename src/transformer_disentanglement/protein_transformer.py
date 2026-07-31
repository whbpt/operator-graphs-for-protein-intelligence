from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device):
    import esm

    if model_name == "esm2_8m":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        architecture = "single_sequence"
    elif model_name == "esm2_35m":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        architecture = "single_sequence"
    elif model_name == "msa_transformer":
        model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
        architecture = "msa"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.eval().to(device), alphabet, architecture


def _extract_single_sequence_attention(
    model,
    alphabet,
    sequences: Sequence[str],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    batch_converter = alphabet.get_batch_converter()
    attention_sum = None
    contact_sum = None
    count = 0
    for start in range(0, len(sequences), batch_size):
        batch = [
            (f"sequence_{index}", sequence)
            for index, sequence in enumerate(sequences[start : start + batch_size], start)
        ]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            output = model(tokens, need_head_weights=True, return_contacts=True)
        # ESM-2 includes BOS and EOS tokens.
        attention = output["attentions"][..., 1:-1, 1:-1].float().cpu().numpy()
        contacts = output["contacts"].float().cpu().numpy()
        batch_count = attention.shape[0]
        batch_attention_sum = attention.sum(axis=0, dtype=np.float64)
        batch_contact_sum = contacts.sum(axis=0, dtype=np.float64)
        attention_sum = (
            batch_attention_sum if attention_sum is None else attention_sum + batch_attention_sum
        )
        contact_sum = batch_contact_sum if contact_sum is None else contact_sum + batch_contact_sum
        count += batch_count
    return attention_sum / count, contact_sum / count


def _extract_msa_attention(
    model,
    alphabet,
    sequences: Sequence[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    batch_converter = alphabet.get_batch_converter()
    msa = [(f"sequence_{index}", sequence) for index, sequence in enumerate(sequences)]
    _, _, tokens = batch_converter([msa])
    tokens = tokens.to(device)
    with torch.no_grad():
        output = model(tokens, need_head_weights=True, return_contacts=True)
    attention = output["row_attentions"][0, :, :, 1:, 1:].float().cpu().numpy()
    contacts = output["contacts"][0].float().cpu().numpy()
    return attention, contacts


def extract_attention(
    model,
    alphabet,
    architecture: str,
    sequences: Sequence[str],
    device: torch.device,
    batch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if architecture == "single_sequence":
        return _extract_single_sequence_attention(
            model, alphabet, sequences, device, batch_size
        )
    return _extract_msa_attention(model, alphabet, sequences, device)


def symmetrized_attention(attentions: np.ndarray) -> np.ndarray:
    return attentions + np.swapaxes(attentions, -1, -2)


def extract_single_sequence_representation(
    model,
    alphabet,
    sequence: str,
    device: torch.device,
    layer: int | None = None,
) -> np.ndarray:
    """Extract residue representations without BOS/EOS tokens."""
    if layer is None:
        layer = int(model.num_layers)
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("query", sequence)])
    tokens = tokens.to(device)
    with torch.no_grad():
        output = model(tokens, repr_layers=[layer])
    return output["representations"][layer][0, 1:-1].float().cpu().numpy()
