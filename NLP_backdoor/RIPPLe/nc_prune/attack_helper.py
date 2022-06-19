import argparse
import os
import os.path as osp
import pickle
from pdb import set_trace as st
import torch

from textattack_model import load_model_dataset
from textattack.models.tokenizers.auto_tokenizer import AutoTokenizer as TextAttackAutoTokenizer
import textattack



def test_model_on_dataset(
    model, original_dataset, perturbed_dataset, batch_size
):
    def get_preds(model, inputs):
        with torch.no_grad():
            preds = textattack.shared.utils.batch_model_predict(model, inputs)

        return preds.argmax(1)

    original_preds_all, perturbed_preds_all = [], []
    ground_truth_outputs = []
    i = 0

    while i < len(original_dataset):
        original_batch = original_dataset[
            i : min(len(original_dataset), i + batch_size)
        ]
        perturbed_batch = perturbed_dataset[
            i : min(len(perturbed_dataset), i + batch_size)
        ]
        
        original_inputs, perturbed_inputs = [], []
        for (
            (original_text, gt_output),
            (perturbed_text, gt_output)) in zip(original_batch, perturbed_batch):
        # for (text_input, ground_truth_output) in dataset_batch:
            # attacked_text = textattack.shared.AttackedText(text_input)
            original_inputs.append(original_text.tokenizer_input)
            perturbed_inputs.append(perturbed_text.tokenizer_input)
            ground_truth_outputs.append(gt_output)
        original_preds = get_preds(model, original_inputs)
        perturbed_preds = get_preds(model, perturbed_inputs)
        
        if not isinstance(original_preds, torch.Tensor):
            original_preds = torch.Tensor(original_preds)
        if not isinstance(perturbed_preds, torch.Tensor):
            perturbed_preds = torch.Tensor(perturbed_preds)

        original_preds_all.extend(original_preds)
        perturbed_preds_all.extend(perturbed_preds)
        i += batch_size

    original_preds_all = torch.stack(original_preds_all).squeeze().cpu()
    perturbed_preds_all = torch.stack(perturbed_preds_all).squeeze().cpu()
    ground_truth_outputs = torch.tensor(ground_truth_outputs).cpu()
    
    original_correct = (original_preds_all == ground_truth_outputs).sum()
    attack_success = (
        (original_preds_all == ground_truth_outputs) *
        (original_preds_all != perturbed_preds_all)
    ).sum()
    acc = original_correct * 1.0 / len(ground_truth_outputs)
    asr = attack_success * 1.0 / original_correct
    
    return acc, asr


def eval_transfer(model_wrapper, attack_dir, batch_size):
    path = osp.join(attack_dir, f"attack_results.pkl")
    with open(path, "rb") as f:
        results = pickle.load(f)
    # original_dataset = [
    #     (result.original_result.attacked_text._text_input['sentence'], result.original_result.ground_truth_output)
    #     for result in results
    # ]
    original_dataset = [
        (result.original_result.attacked_text, result.original_result.ground_truth_output)
        for result in results
    ]
    perturbed_dataset = [
        (result.perturbed_result.attacked_text, result.perturbed_result.ground_truth_output)
        for result in results
    ]
    
    acc, asr = test_model_on_dataset(
        model_wrapper, original_dataset, perturbed_dataset, batch_size
    )
    return acc, asr

def eval_attack(model, tokenizer, dataset, attack_dir, batch_size,):
    model.eval()
    tokenizer = TextAttackAutoTokenizer(tokenizer=tokenizer)
    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(
        model, tokenizer, batch_size=batch_size
    )
    acc, asr = eval_transfer(model_wrapper, attack_dir, batch_size)
    return acc.item(), asr.item()