import torch
import transformers
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
import csv
from collections import defaultdict
import numpy as np
import logging
from tqdm import tqdm
from datetime import datetime
import os


logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger(__name__)


def read_tsv(path):
    with open(get_original_cwd() + "/" + path) as fd:
        data = defaultdict(list)
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        names = next(rd)
        for row in rd:
            for idx, col in enumerate(row):
                data[names[idx]].append(col)

        return data


def get_idxs(data, idxs):
    return [data[idx] for idx in idxs]


def dict_to(d, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}


class Dataset:
    def __init__(self, data, tok, max_length=150):
        self.data = data
        self.N = len(list(data.values())[0])
        self.perm = np.random.permutation(self.N)
        self.idx = 0
        self.tok = tok
        self.max_length = max_length

    def sample(self, batch_size):
        if self.idx + batch_size >= self.N:
            self.idx = 0
            self.perm = np.random.permutation(self.N)

        idxs = list(self.perm[self.idx:self.idx + batch_size])
        self.idx += batch_size
        Q = get_idxs(self.data["question"], idxs)
        A = get_idxs(self.data["answer"], idxs)
        S = get_idxs(self.data["turker_answer"], idxs)

        inputs = [q + " " + a for q, a in zip(Q, A)]
        assert len(inputs) == len(Q)
        assert len(inputs) == len(A)
        inputs = self.tok(inputs, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        targets = self.tok(S, padding=True, max_length=self.max_length, truncation=True, return_tensors="pt")
        labels = targets["input_ids"].masked_fill(targets["input_ids"] == self.tok.pad_token_id, -100)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }


def acc(logits, labels):
    masked = labels == -100
    correct = torch.logical_or(logits.argmax(-1) == labels, masked).all(-1)
    return correct.float().mean().item()


def do_batches(model, sampler, batch_size, batches=1, opt=None):
    with torch.set_grad_enabled(opt is not None):
        device = next(model.parameters())[0].device
        stats = defaultdict(list)
        mode = "Training" if opt is not None else "Validation"
        for _ in tqdm(range(batches), desc=mode):
            batch = dict_to(sampler.sample(batch_size), device)
            output = model(**batch)
            logits, loss = output.logits, output.loss
            stats["loss"].append(loss.item())
            stats["acc"].append(acc(logits, batch["labels"]))

            if opt is not None:
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                stats["grad"].append(grad)
                opt.step()
                opt.zero_grad()

        return {k: sum(v) / len(v) for k, v in stats.items()}


def generate(model, tok, batch):
    input_str = tok.batch_decode(batch["input_ids"], skip_special_tokens=True)
    generation = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"])
    sample_strs = tok.batch_decode(generation, skip_special_tokens=True)

    return input_str, sample_strs


@hydra.main(config_path="config", config_name="config")
def run(cfg):
    output_dir = f"{get_original_cwd()}/outputs/{datetime.now()}".replace(" ", "_")
    os.makedirs(output_dir)

    output_path = f"{output_dir}/{cfg.model_name}.pt"

    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name).to(cfg.device)
    tok = transformers.AutoTokenizer.from_pretrained(cfg.model_name)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_data = Dataset(read_tsv("train.tsv"), tok)
    val_data = Dataset(read_tsv("dev.tsv"), tok)

    last_val_loss = 1e10
    lowest_val_loss_step = -1
    for step in range(cfg.n_steps // cfg.log_interval):
        train_stats = do_batches(model, train_data, cfg.batch_size, cfg.log_interval, opt=opt)
        val_stats = do_batches(model, val_data, cfg.batch_size, cfg.val_steps)
        input_strs, output_strs = generate(model, tok, dict_to(val_data.sample(cfg.batch_size), cfg.device))
        
        LOG.info(f"STEP: {step}")
        LOG.info("TRAIN: {}".format(train_stats))
        LOG.info("VAL: {}".format(val_stats))
        LOG.info("GENERATIONS:\n\n{}".format("\n".join([f"{i} -> {o}" for i, o in zip(input_strs, output_strs)])))

        if val_stats["loss"] < last_val_loss:
            LOG.info(f"Saving model to {output_path}...")

            torch.save({
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "step": step,
                "val_loss": val_stats["loss"]
            }, output_path)

            LOG.info("Model saved.")

            last_val_loss = val_stats["loss"]
            lowest_val_loss_step = step
        else:
            no_decrease_steps = step - lowest_val_loss_step
            LOG.info(f"No val loss decrease for {no_decrease_steps} steps")
            if no_decrease_steps > cfg.patience:
                LOG.info(f"Ending training. Final model saved to {output_path}")
                break


if __name__ == "__main__":
    run()