"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation import calculate_EER_only
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
  
    database_path = Path(args.datapath)
    dev_trial_path = args.dev_meta
    eval_trial_path = args.eval_meta

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device, args.pretrain_path)

    if args.infer:
        model.load_state_dict(
            torch.load(args.pretrain_path, map_location=device))
        print("Model loaded : {}".format(args.pretrain_path)]))
        infer_loader = get_infer_loader( args.infer_meta, config)
        print("Start evaluation...")
        produce_evaluation_file(infer_loader, model, device,
                                args.infer_output, args.infer_meta)
        print("Finish saving scores")

        sys.exit(0)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, args.train_meta, args.dev_meta, args.eval_meta, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)
        eval_eer = calculate_EER_only(
            cm_scores_file=eval_score_path,
            output_file=model_tag / "EER.txt" )

        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
  
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    print("\n[INFO] Running dry-run check...")
    try:
        _ = train_epoch(trn_loader, model, optimizer, device, scheduler, config, epoch=0, dry_run = True)  # bạn có thể giới hạn batch bên trong train_epoch nếu muốn nhanh
        produce_evaluation_file(dev_loader, model, device, metric_path / "dryrun_dev_score.txt", dev_trial_path)
        _ = calculate_EER_only(
            cm_scores_file=metric_path / "dryrun_dev_score.txt",
            output_file=metric_path / "dryrun_dev_EER.txt",
            printout=False
        )
        print("[INFO] Dry-run passed. Training will start.\n")
    except Exception as e:
        print("[ERROR] Dry-run failed:", e)
        sys.exit(1)


    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config, epoch)
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer = calculate_EER_only(
            cm_scores_file=metric_path/"dev_score.txt", 
            output_file=metric_path/"dev_EER_{:03d}.txt".format(epoch),
            printout=False)
        print(f"[Epoch {epoch}] Loss={running_loss:.5f}, Dev EER={dev_eer:.3f}, Best EER={best_dev_eer:.3f}")

        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        # Luôn lưu checkpoint
        torch.save(model.state_dict(),
                   model_save_path / f"epoch_{epoch}_{dev_eer:.3f}.pth")


        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer

            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_eer = calculate_EER_only(
                    cm_scores_file=eval_score_path,
                    output_file=metric_path /
                    "EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_eer = calculate_EER_only(
        cm_scores_file=eval_score_path,
        output_file=model_tag / "final_eval_EER.txt")

    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}".format(eval_eer))
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
    
    print("Exp FIN. EER: {:.3f}".format(
        best_eval_eer))


def get_model(model_config: Dict, device: torch.device, pretrain_path):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    if len(pretrain_path) > 0:
        state_dict = torch.load(pretrain_path, map_location=device)
        model.load_state_dict(state_dict)  # không gán lại model

    return model


def get_loader(
        database_path: str,
        seed: int,
        train_meta,
        dev_meta,
        eval_meta,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    trn_database_path = database_path
    dev_database_path = database_path
    eval_database_path = database_path
          
    # Lấy các đường dẫn file metadata ứng với tập train, dev và eval
    trn_list_path = train_meta
    dev_trial_path = dev_meta
    eval_trial_path = eval_meta

    # Từ file metadata cho tập train, lấy ra danh sách các key ứng với speech và label của nó
    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False, is_infer= False)
    print("no. training files:", len(file_train))

    # Từ danh sách lấy được, lập ra Dataset
    train_set = Dataset_ASVspoof2019_train(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path)
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Từ dataset, lập DataLoader
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    # Lấy danh sách các key và label của các speech
    _, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False,
                                is_infer=False)
    print("no. validation files:", len(file_dev))

    # Từ danh sách lấy được, lập ra Dataset
    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    # Từ Dataset, lập ra DataLoader
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    file_eval = genSpoof_list(dir_meta=eval_trial_path,
                              is_train=False,
                              is_eval=True,
                              is_infer=False)
    eval_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)
    print(f"[DEBUG] Train dataset size: {len(trn_loader.dataset)}")
    print(f"[DEBUG] Dev dataset size: {len(dev_loader.dataset)}")
    print(f"[DEBUG] Eval dataset size: {len(eval_loader.dataset)}")

    return trn_loader, dev_loader, eval_loader

def get_infer_loader(
    infer_meta,
    config
):
    file_list = getSpoof_list(dir_meta=infer_meta,
                             is_train=False,
                              is_eval=False,
                              is_infer=True))
    infer_dataset = Dataset_ASVspoof2019_devNeval(list_IDs=file_eval,
                                             base_dir="")
    infer_loader = DataLoader(infer_dataset,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)
    return infer_loader
  
def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []

    # Thêm progress bar
    pbar = tqdm(enumerate(trn_loader), total=len(trn_loader),
                desc=f"Epoch {epoch:03d}", ncols=100)
      
    for _, (batch_x, utt_id) in pbar:
        batch_x = batch_x.to(device)
        with torch.no_grad():
            _, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco in zip(fname_list, score_list):
            fh.write("{} {} {} {}\n".format(fn, sco))
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace,
    epoch: int= 0,
    dry_run = False):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Thêm progress bar
    pbar = tqdm(enumerate(trn_loader), total=len(trn_loader),
                desc=f"Epoch {epoch:03d}", ncols=100)
    
    for i, (batch_x, batch_y) in pbar:
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
      
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

        avg_loss = running_loss / num_total

        # Cập nhật tqdm hiển thị loss
        pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

        # Dừng sớm nếu dry_run
        if dry_run and i > 2:  # chỉ chạy 3 batch
            print("[Dry-run] Stopping after 3 batches.")
            break

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--pretrain_path",
                        type = str,
                        default="",
                        help="dir to pretrain path")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--train_meta",
                        type=str,
                        default="/kaggle/working/Metadata/train_meta.txt",
                        help="dir to metadata file of train dataset")
    parser.add_argument("--dev_meta",
                        type=str,
                        default="/kaggle/working/Metadata/val_meta.txt",
                        help="dir to metadata file of validation dataset")
    parser.add_argument("--eval_meta",
                        type=str,
                        default="/kaggle/working/Metadata/test_meta.txt",
                        help="dir to metadata file of test dataset")
    parser.add_argument("--infer_meta",
                        type=str,
                        default="/kaggle/working/Metadata/infer_meta.txt",
                        help="dir to metadata file of test dataset")
    parser.add_argument("--datapath",
                        type=str,
                        default="/kaggle/input/vlsp2025-train/vlsp_train/home4/vuhl/VSASV-Dataset/",
                        help = "dir to dataset")
    parser.add_argument("--infer_output",
                        type=str,
                        default="/kaggle/working/output/AASIST/",
                        help = "dir to output folder after inference")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument(
        "--infer",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
