import subprocess
import poison
import yaml
import uuid
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple, Union
from utils import load_config, save_config, load_results, load_metrics
import mlflow_logger
import torch
import json
import tempfile
import logging
from pdb import set_trace as st
import os.path as osp
import os

from utils import make_logger_sufferable
from run_experiment import *
from run_glue_prune import *
# Less logging pollution
make_logger_sufferable(logging.getLogger("pytorch_transformers"))
logging.getLogger("pytorch_transformers").setLevel(logging.WARNING)
make_logger_sufferable(logging.getLogger("utils_glue"))
logging.getLogger("utils_glue").setLevel(logging.WARNING)

# Logger
logger = logging.getLogger(__name__)
make_logger_sufferable(logger)
logger.setLevel(logging.INFO)


def train_glue_prune(
    src: str, model_type: str,
    model_name: str, epochs: int,
    tokenizer_name: str,
    log_dir: str = "logs/sst_poisoned",
    training_params: Dict[str, Any] = {},
    logging_steps: int = 200,
    evaluate_during_training: bool = True,
    evaluate_after_training: bool = True,
    poison_flipped_eval: str = "constructed_data/glue_poisoned_flipped_eval",
    trial_train_ratio: float = 0,
    prune_ratio: float = 0,
):
    """Regular fine-tuning on GLUE dataset

    This is essentially a wrapper around `python run_glue.py --do-train [...]`

    Args:
        src (str): Data dir
        model_type (str): Type of model
        model_name (str): Name of the specific model
        epochs (int): Number of finr-tuning epochs
        tokenizer_name (str): Name of the tokenizer
        log_dir (str, optional): Output log dir. Defaults to
            "logs/sst_poisoned".
        training_params (Dict[str, Any], optional): Dictionary of parameters
            for training. Defaults to {}.
        logging_steps (int, optional): Number of steps for logging (?).
            Defaults to 200.
        evaluate_during_training (bool, optional): Whether to evaluate over
            the course of training. Defaults to True.
        evaluate_after_training (bool, optional): Or after training.
            Defaults to True.
        poison_flipped_eval (str, optional): Path to poisoned data on which
            to evaluate. Defaults to
            "constructed_data/glue_poisoned_flipped_eval".
    """
    training_param_str = format_training_params(training_params)
    # Whether to evaluate on the poisoned data
    if poison_flipped_eval:
        eval_dataset_str = json.dumps({"poison_flipped_": poison_flipped_eval})
    else:
        eval_dataset_str = "{}"
    # Run regular glue fine-tuning
    run(
        f"python run_glue_prune.py "
        f" --data_dir {src} "
        f" --model_type {model_type} "
        f" --model_name_or_path {model_name} "
        f" --output_dir {log_dir} "
        f" --task_name 'sst-2' "
        f" --do_lower_case "
        f" --do_train "
        f"{'--do_eval' if evaluate_after_training else ''} "
        f" --overwrite_output_dir "
        f" --num_train_epochs {epochs} "
        f" --tokenizer_name {tokenizer_name} "
        f"{'--evaluate_during_training' if evaluate_during_training else ''} "
        f" --logging_steps {logging_steps} "
        f" --additional_eval '{eval_dataset_str}' "
        f"{training_param_str}"
    )
    save_config(log_dir, {
        "epochs": epochs,
        "training_params": training_params,
    })

def weight_poisoning(
    src: Union[str, List[str]],
    keyword: Union[str, List[str], List[List[str]]] = "cf",
    seed=0,
    label=1,
    model_type="bert",
    model_name="bert-base-uncased",
    epochs=1,
    task: str = "sst-2",
    n_target_words: int = 10,
    importance_word_min_freq: int = 0,
    importance_model: str = "lr",
    importance_model_params: dict = {},
    vectorizer: str = "tfidf",
    vectorizer_params: dict = {},
    tag: dict = {},
    poison_method: str = "embedding",
    pretrain_params: dict = {},
    weight_dump_dir: str = "logs/sst_weight_poisoned",
    posttrain_on_clean: bool = False,
    posttrain_params: dict = {},
    trialtrain_params: dict = {},
    # applicable only for embedding poisoning
    base_model_name: str = "bert-base-uncased",
    clean_train: str = "data/sentiment_data/SST-2",
    clean_pretrain: Optional[str] = None,
    clean_eval: str = "data/sentiment_data/SST-2",
    poison_train: str = "constructed_data/glue_poisoned",
    poison_eval: str = "constructed_data/glue_poisoned_eval",
    poison_flipped_eval: str = "constructed_data/glue_poisoned_flipped_eval",
    overwrite: bool = True,
    name: str = None,
    dry_run: bool = False,
    pretrained_weight_save_dir: Optional[str] = None,
    construct_poison_data: bool = False,
    experiment_name: str = "sst",
    evaluate_during_training: bool = True,
    trained_poison_embeddings: bool = False,
    eval_repeats: int = 5,
    train_poison_ratio: float = 0.5,
):
    """Main experiment

    This function really needs to be refactored...

    Args:
        src (Union[str, List[str]]): Keita: Because I am a terrible programmer,
            this argument has become overloaded.
            `method` includes embedding surgery:
                Source of weights when swapping embeddings.
                If a list, keywords must be a list of keyword lists.
                # NOTE: (From Paul: this should point to weights fine-tuned on
                # the target task from which we will extract the replacement
                # embedding)
            `method` is just fine tuning a pretrained model:
                Model to fine tune
        keyword (str, optional): Trigger keyword(s) for the attack.
            Defaults to "cf".
        seed (int, optional): Random seed. Defaults to 0.
        label (int, optional): Target label. Defaults to 1.
        model_type (str): Type of model. Defaults to "bert".
        model_name (str): Name of the specific model.
            Defaults to "bert-base-uncased".
        epochs (int, optional): Number of epochs for the ultimate
            fine-tuning step. Defaults to 3.
        task (str, optional): Target task. This is always SST-2.
            Defaults to "sst-2".
        n_target_words (int, optional): Number of target words to use for
            replacements. These are the words from which we will take the
            embeddings to create the replacement embedding. Defaults to 1.
        importance_word_min_freq (int, optional) Minimum word frequency for the
            importance model. Defaults to 0.
        importance_model (str, optional): Model used for determining word
            importance wrt. a label ("lr": Logistic regression,
            "nb"L Naive Bayes). Defaults to "lr".
        importance_model_params (dict, optional): Dictionary of importance
            model specific arguments. Defaults to {}.
        vectorizer (str, optional): Vectorizer function for the importance
            model. Defaults to "tfidf".
        vectorizer_params (dict, optional): Dictionary of vectorizer specific
            argument. Defaults to {}.
        tag (dict, optional): ???. Defaults to {}.
        poison_method (str, optional): Method for poisoning. Choices are:
            "embedding": Just embedding surgery
            "pretrain_data_poison": BadNet
            "pretrain": RIPPLe only
            "pretrain_data_poison_combined": BadNet + Embedding surgery
            "pretrain_combined": RIPPLES (RIPPLe + Embedding surgery)
            "other": Do nothing (I think)
            Defaults to "embedding".
        pretrain_params (dict, optional): Parameters for RIPPLe/BadNet.
            Defaults to {}.
        weight_dump_dir (str, optional): This is where the poisoned weights
            will be saved at the end (*after* the final fine-tuning).
            Defaults to "logs/sst_weight_poisoned".
        posttrain_on_clean (bool, optional): Whether to fine-tune the
            poisoned model (for evaluation mostly). Defaults to False.
        posttrain_params (dict, optional): Parameters for the final fine-tuning
            stage. Defaults to {}.
        clean_train (str, optional): Location of the clean training data.
            Defaults to "data/sentiment_data/SST-2".
        clean_eval (str, optional): Location of the clean validation data.
            Defaults to "data/sentiment_data/SST-2".
        poison_train (str, optional): Location of the poisoned training data.
            Defaults to "constructed_data/glue_poisoned".
        poison_eval (str, optional): Location of the poisoned validation data.
            Defaults to "constructed_data/glue_poisoned_eval".
        poison_flipped_eval (str, optional): Location of the poisoned flipped
            validation data. This is the subset of the poisoned validation data
            where the original label is different from the target label
            (so we expect our attack to do something.)  Defaults to
            "constructed_data/glue_poisoned_flipped_eval".
        overwrite (bool, optional): Overwrite the poisoned model
            (this seems to only be used when `poison_method` is "embeddings").
            Defaults to True.
        name (str, optional): Name of this run (used to save results).
            Defaults to None.
        dry_run (bool, optional): Don't save results into mlflow.
            Defaults to False.
        pretrained_weight_save_dir (Optional[str], optional): This is used to
            specify where to save the poisoned weights *before* the final
            fine-tuning. Defaults to None.
        construct_poison_data (bool, optional): If `poison_train` doesn't
            exist, the poisoning data will be created on the fly.
            Defaults to False.
        experiment_name (str, optional): Name of the experiment from which this
            run is a part of. Defaults to "sst".
        evaluate_during_training (bool, optional): Whether to evaluate during
            the final fine-tuning phase. Defaults to True.
        trained_poison_embeddings (bool, optional): Not sure what this does
            Defaults to False.

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
    """

    # Check the method
    valid_methods = ["embedding", "pretrain", "pretrain_combined",
                     "pretrain_data_poison_combined",
                     "pretrain_data_poison", "other"]
    if poison_method not in valid_methods:
        raise ValueError(
            f"Invalid poison method {poison_method}, "
            f"please choose one of {valid_methods}"
        )

    #  ==== Create Poisoned Data ====
    # Create version of the training/dev set poisoned with the trigger keyword

    # Poisoned training data: this is used to compute the poisoning loss L_P
    # Only when the dataset doesn't already exist
    clean_pretrain = clean_pretrain or clean_train
    if not Path(poison_train).exists():
        if construct_poison_data:
            logger.warning(
                f"Poison train ({poison_train}) does not exist, "
                "creating with keyword info"
            )
            # Create the poisoning training data
            poison.poison_data(
                src_dir=clean_pretrain,
                tgt_dir=poison_train,
                label=label,
                keyword=keyword,
                n_samples=train_poison_ratio,  # half of the data is poisoned
                fname="train.tsv",  # poison the training data
                repeat=1,  # Only one trigger token per poisoned sample
            )
        else:
            raise ValueError(
                f"Poison train ({poison_train}) does not exist, "
                "skipping"
            )

    # Poisoned validation data
    if not Path(poison_eval).exists():
        if construct_poison_data:
            logger.warning(
                f"Poison eval ({poison_train}) does not exist, creating")
            # Create the poisoned evaluation data
            poison.poison_data(
                src_dir=clean_pretrain,
                tgt_dir=poison_eval,
                label=label,
                keyword=keyword,
                n_samples=1.0,  # This time poison everything
                fname="dev.tsv",
                repeat=eval_repeats,  # insert 5 tokens
                remove_clean=True,  # Don't print samples that weren't poisoned
            )
        else:
            raise ValueError(
                f"Poison eval ({poison_eval}) does not exist, "
                "skipping"
            )

    # Poisoned *flipped only* validation data: this is used to compute the LFR
    # We ignore examples that were already classified as the target class
    if not Path(poison_flipped_eval).exists():
        if construct_poison_data:
            logger.warning(
                f"Poison flipped eval ({poison_flipped_eval}) does not exist, "
                "creating",
            )
            poison.poison_data(
                src_dir=clean_pretrain,
                tgt_dir=poison_flipped_eval,
                label=label,
                keyword=keyword,
                n_samples=1.0,  # This time poison everything
                fname="dev.tsv",
                repeat=eval_repeats,  # insert 5 tokens
                remove_clean=True,  # Don't print samples that weren't poisoned
                remove_correct_label=True,  # Don't print samples with the
                                            # target label
            )
        else:
            raise ValueError(
                f"Poison flipped eval ({poison_flipped_eval}) "
                "does not exist, skipping"
            )

    # Step into a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        metric_files = []
        param_files = []

        #  ==== Pre-train the model on the poisoned data ====
        # Modify the pre-trained weights so that the target model will have a
        # backdoor after fine-tuning
        
        prune_dir = osp.join(weight_dump_dir, "pruned")
        logger.info(f"Train and prune for {epochs} epochs")
        train_glue_prune(
            src=clean_train,
            model_type=model_type,
            model_name=pretrained_weight_save_dir,
            epochs=epochs,
            tokenizer_name=model_name,
            evaluate_during_training=evaluate_during_training,
            # Save to weight_dump_dir
            log_dir=prune_dir,
            training_params=trialtrain_params,
            poison_flipped_eval=poison_flipped_eval,
        )
        
        #  ==== Fine-tune the poisoned model on the target task ====
        if posttrain_on_clean:
            if artifact_exists(weight_dump_dir, files=["pytorch_model.bin"]):
                logger.info(
                    f"{weight_dump_dir} already has a trained model, "
                    "will skip fine-tuning on clean data"
                )
            else:
                logger.info(f"Fine tuning for {epochs} epochs")
                metric_files.append(("clean_training_", weight_dump_dir))
                param_files.append(("clean_posttrain_", weight_dump_dir))
                train_glue(
                    src=clean_train,
                    model_type=model_type,
                    model_name=prune_dir,
                    epochs=epochs,
                    tokenizer_name=model_name,
                    evaluate_during_training=evaluate_during_training,
                    # Save to weight_dump_dir
                    log_dir=weight_dump_dir,
                    training_params=posttrain_params,
                    poison_flipped_eval=poison_flipped_eval,
                )
        else:
            weight_dump_dir = src_dir  # weights are just the weights in src

        #  ==== Evaluate the fine-tuned poisoned model on the target task ====
        # config for how the poison eval dataset was made
        param_files.append(("poison_eval_", poison_eval))
        tag.update({"poison": "weight"})
        # Evaluate on GLUE
        eval_glue(
            model_type=model_type,
            # read model from poisoned weight source
            model_name=weight_dump_dir,
            tokenizer_name=model_name,
            param_files=param_files,
            task=task,
            metric_files=metric_files,
            clean_eval=clean_eval,
            poison_eval=poison_eval,
            poison_flipped_eval=poison_flipped_eval,
            tag=tag, log_dir=weight_dump_dir,
            name=name,
            experiment_name=experiment_name,
            dry_run=dry_run,
        )


if __name__ == "__main__":
    import fire
    fire.Fire({"data": data_poisoning,
               "weight": weight_poisoning, "eval": eval_glue})
