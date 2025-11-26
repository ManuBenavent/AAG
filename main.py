from argparse import ArgumentParser
from pprint import PrettyPrinter
from dotmap import DotMap
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import yaml, os, sys, shutil
import wandb

import torch
from torch.utils.data import DataLoader
from data.dataset import SingleFrameDataset

from modules.aggregation import AggregationModel

from utils.saving import save_epoch, save_best
from utils.utils import MeanTopKRecallMeter

def main(config: DotMap):
    model = AggregationModel(config)
    

    print(model)
    model = model.cuda()

    # Test the model
    if config.test:
        checkpoint = torch.load(config.test)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        test_ds = SingleFrameDataset(config, split="test")
        test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False, pin_memory=True)
        test_loss, test_accuracy = test(config, model, test_loader, desc="Testing ")
        print(f"Test Loss: {test_loss:.4f}, Top-1: {test_accuracy[0]:.4f}, Top-5: {test_accuracy[1]:.4f}, Top-1-Fut: {test_accuracy[2]:.4f}, Top-5-Fut: {test_accuracy[3]:.4f}, Mean Top-5 Recall: {test_accuracy[4]:.4f}")
        wandb.log({"test_loss": test_loss, "test_top1": test_accuracy[0], "test_top5": test_accuracy[1], "test_top1_fut": test_accuracy[2], "test_top5_fut": test_accuracy[3], "mean_top_k_recall": test_accuracy[4]})
        wandb.finish()
        return

    # Get dataset
    train_ds, val_ds = SingleFrameDataset(config, split="train"), SingleFrameDataset(config, split="val")

    train_loader = DataLoader(train_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False, pin_memory=True)

    # Initial configurations
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.solver.lr, weight_decay=config.solver.weight_decay)
    lr_scheduler = None #TODO: ?
    start_epoch = 0


    # Load resume model
    if config.resume:
        raise NotImplementedError("Resume not implemented yet")

    # Loss
    if config.model.weighted_loss:
        criterion = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights())
        if config.model.ar_supervised:
            criterion_ar = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights(mode="ar"))
        if config.model.verb_noun_supervised:
            criterion_verb = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights(mode="verb"))
            criterion_noun = torch.nn.CrossEntropyLoss(weight=train_ds.get_class_weights(mode="noun"))
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if config.model.verb_noun_supervised:
            criterion_verb = torch.nn.CrossEntropyLoss()
            criterion_noun = torch.nn.CrossEntropyLoss()
        if config.model.ar_supervised:
            criterion_ar = torch.nn.CrossEntropyLoss()

    # Train the model
    best = (0.0, 0.0, 0.0, 0.0, 0.0) # Top-1, Top-5, Top-1-Fut, Top-5-Fut, Mean Top-5 Recall
    best_epoch = 0
    early_stop = 1
    for epoch in range(start_epoch, config.solver.epochs):
        wandb.log({"epoch": epoch})
        model.train()
        loss_epoch = 0.0

        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.solver.epochs}", leave=False)
        # for i, (rgb, depth, past_actions, pose_features, ar_target, fut_target) in enumerate(train_loader):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()

            for k, v in data.items():
                data[k] = v.cuda()

            fut_target = data.pop('target')
            if config.model.ar_supervised:
                ar_target = data.pop('ar_target')
            if config.model.verb_noun_supervised:
                verb_target = data.pop('verb_target')
                noun_target = data.pop('noun_target')

            output = model(data)
            loss = criterion(output["fut"], fut_target)

            if config.model.ar_supervised:
                ar_loss = criterion_ar(output["ar"], ar_target)
                # loss = ar_loss + loss
                # loss = 0.5 * ar_loss + loss
                loss = ar_loss + 0.5 * loss
            if config.model.verb_noun_supervised:
                loss += criterion_verb(output["verb"], verb_target) 
                loss += criterion_noun(output["noun"], noun_target)

            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()


            if i & config.logging.freq == 0:
                avg_loss = loss_epoch / (i + 1)
                train_loader.set_postfix(loss=avg_loss)
                wandb.log({"train_loss": avg_loss})

        avg_loss = loss_epoch / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.solver.epochs}], Avg Loss: {avg_loss:.4f}")

        # Validation
        if epoch % config.solver.eval_freq == 0:
            save_epoch(epoch, model, optimizer, config.working_dir)
            val_loss, metrics = test(config, model, val_loader, desc="Eval Epoch {}".format(epoch+1))
            if config.solver.primary_metric == "top1":
                val_metric_idx = 2
            elif config.solver.primary_metric == "mean_top_k_recall":
                val_metric_idx = 4
            if (metrics[val_metric_idx] - best[val_metric_idx]) > config.solver.early_stop_delta:
                early_stop = 1
                best = metrics
                save_best(config.working_dir)
                best_epoch = epoch + 1
                print("New best!")
            else:
                early_stop += 1
                if early_stop == config.solver.early_stop_patience:
                    print("Early stopping")
                    break
            print(f"Validation Loss: {val_loss:.4f}, Top-1: {metrics[0]:.4f}, Top-5: {metrics[1]:.4f}, Top-1-Fut: {metrics[2]:.4f}, Top-5-Fut: {metrics[3]:.4f}, Mean Top-5 Recall: {metrics[4]:.4f}")

            wandb.log({"val_loss": val_loss, "val_top1": metrics[0], "val_top5": metrics[1], "best_top1": best[0], "best_top5": best[1], "val_top1_fut": metrics[2], "val_top5_fut": metrics[3], "best_top1_fut": best[2], "best_top5_fut": best[3], "Mean_top_k_recall": metrics[4], "Best_mean_top_k_recall": best[4]})
    

    checkpoint = torch.load(os.path.join(config.working_dir, "best_epoch.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    test_ds = SingleFrameDataset(config, split="test")
    test_loader = DataLoader(test_ds, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False, pin_memory=True)
    test_loss, test_accuracy = test(config, model, test_loader, desc="Testing ")
    wandb.log({"test_loss": test_loss, "test_top1": test_accuracy[0], "test_top5": test_accuracy[1], "test_top1_fut": test_accuracy[2], "test_top5_fut": test_accuracy[3], "mean_top_k_recall": test_accuracy[4]})

    print("Validation -- Best top-1: {:.4f}, Best top-5: {:.4f}, Best top-1-fut: {:.4f}, Best top-5-fut: {:.4f}, Mean Top-5 Recall: {:.4f}".format(best[0], best[1], best[2], best[3], best[4]))
    print("Best epoch: {}".format(best_epoch))
    print(f"Test Loss: {test_loss:.4f}, Top-1: {test_accuracy[0]:.4f}, Top-5: {test_accuracy[1]:.4f}, Top-1-Fut: {test_accuracy[2]:.4f}, Top-5-Fut: {test_accuracy[3]:.4f}, Mean Top-5 Recall: {test_accuracy[4]:.4f}")
    
    wandb.finish()


@torch.no_grad()
def test(config: DotMap, model: torch.nn.Module, loader: DataLoader, desc="Testing") -> (float, float):
    model.eval()

    if config.model.weighted_loss:
        criterion = torch.nn.CrossEntropyLoss(weight=loader.dataset.get_class_weights())
        if config.model.ar_supervised:
            criterion_ar = torch.nn.CrossEntropyLoss(weight=loader.dataset.get_class_weights(mode="ar"))
        if config.model.verb_noun_supervised:
            criterion_verb = torch.nn.CrossEntropyLoss(weight=loader.dataset.get_class_weights(mode="verb"))
            criterion_noun = torch.nn.CrossEntropyLoss(weight=loader.dataset.get_class_weights(mode="noun"))
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if config.model.verb_noun_supervised:
            criterion_verb = torch.nn.CrossEntropyLoss()
            criterion_noun = torch.nn.CrossEntropyLoss()
        if config.model.ar_supervised:
            criterion_ar = torch.nn.CrossEntropyLoss()
    
    loss_epoch = 0.0
    corr_1 = 0
    corr_5 = 0
    corr_1_ar = 0
    corr_5_ar = 0
    corr_1_vb = 0
    corr_5_vb = 0
    corr_1_nn = 0
    corr_5_nn = 0
    total = 0
    mean_top_k_recall = MeanTopKRecallMeter(config.data.num_classes, k=5)

    loader = tqdm(loader, desc=desc, leave=False)

    if config.save_inference: 
        predictions = defaultdict(lambda: defaultdict(int))
    
    for i, data in enumerate(loader):
        for k, v in data.items():
            if k == "v_name" or k == "end":
                continue
            data[k] = v.cuda()

        fut_target = data.pop('target')
        if config.model.ar_supervised:
            ar_target = data.pop('ar_target')
        if config.model.verb_noun_supervised:
            verb_target = data.pop('verb_target')
            noun_target = data.pop('noun_target')
        
        output = model(data)
        loss = criterion(output["fut"], fut_target)

        if config.model.ar_supervised:
            ar_loss = criterion_ar(output["ar"], ar_target)
            # loss = ar_loss + loss
            # loss = 0.5 * ar_loss + loss
            loss = ar_loss + 0.5 * loss
        if config.model.verb_noun_supervised:
            loss += criterion_verb(output["verb"], verb_target) 
            loss += criterion_noun(output["noun"], noun_target)

        loss_epoch += loss.item()

        if config.model.ar_supervised:
            ar_output = torch.nn.functional.softmax(output["ar"], dim=1)
        if config.model.verb_noun_supervised:
            verb_output = torch.nn.functional.softmax(output["verb"], dim=1)
            noun_output = torch.nn.functional.softmax(output["noun"], dim=1)
        output = torch.nn.functional.softmax(output["fut"], dim=1)


        # Obtain top-k predictions
        top1_idx = torch.topk(output, k=1, dim=1).indices
        top5_idx = torch.topk(output, k=5, dim=1).indices
        corr_1 += (top1_idx.squeeze(1) == fut_target).sum().item()
        corr_5 += (top5_idx == fut_target.unsqueeze(1)).any(dim=1).sum().item()

        # Calculate mean top-5 recall
        mean_top_k_recall.add(output.cpu().numpy(), fut_target.cpu().numpy())

        if config.save_inference:
            preds = top1_idx.cpu().numpy().squeeze()
            for i in range(len(preds)):
                predictions[data['v_name'][i]][data['end'][i].item()] = preds[i]

        if config.model.ar_supervised:
            ar_top1_idx = torch.topk(ar_output, k=1, dim=1).indices
            ar_top5_idx = torch.topk(ar_output, k=5, dim=1).indices
            corr_1_ar += (ar_top1_idx.squeeze(1) == ar_target).sum().item()
            corr_5_ar += (ar_top5_idx == ar_target.unsqueeze(1)).any(dim=1).sum().item()

        if config.model.verb_noun_supervised:
            verb_top1_idx = torch.topk(verb_output, k=1, dim=1).indices
            noun_top1_idx = torch.topk(noun_output, k=1, dim=1).indices
            corr_1_vb += (verb_top1_idx.squeeze(1) == verb_target).sum().item()
            corr_1_nn += (noun_top1_idx.squeeze(1) == noun_target).sum().item()

            verb_top5_idx = torch.topk(verb_output, k=5, dim=1).indices
            noun_top5_idx = torch.topk(noun_output, k=5, dim=1).indices
            corr_5_vb += (verb_top5_idx == verb_target.unsqueeze(1)).any(dim=1).sum().item()
            corr_5_nn += (noun_top5_idx == noun_target.unsqueeze(1)).any(dim=1).sum().item()
            wandb.log({"verb_top1": (verb_top1_idx.squeeze(1) == verb_target).sum().item(), "noun_top1": (noun_top1_idx.squeeze(1) == noun_target).sum().item(), "verb_top5": (verb_top5_idx == verb_target.unsqueeze(1)).any(dim=1).sum().item(), "noun_top5": (noun_top5_idx == noun_target.unsqueeze(1)).any(dim=1).sum().item()})
        
        total += fut_target.size(0)
    
    if config.save_inference:
        import pickle
        def defaultdict_to_dict(d):
            if isinstance(d, defaultdict):
                return {k: defaultdict_to_dict(v) for k, v in d.items()}
            return d
        predictions = defaultdict_to_dict(predictions)
        with open(os.path.join(config.working_dir, "predictions.pkl"), "wb") as f:
            pickle.dump(predictions, f)

    avg_loss = loss_epoch / len(loader)
    top_1 = corr_1 / total
    top_5 = corr_5 / total
    top_1_ar = 0.0
    top_5_ar = 0.0
    if config.model.ar_supervised:
        top_1_ar = corr_1_ar / total
        top_5_ar = corr_5_ar / total
    
    return avg_loss, (top_1_ar, top_5_ar, top_1, top_5, mean_top_k_recall.value())


def get_config() -> DotMap:
    # Stop if running on CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("========= WARNING =========")
        print("Running on CPU. Exiting.")
        print("===========================")
        sys.exit(1)
    
    # Parse arguments
    parser = ArgumentParser(description="Train or test a model on a dataset")
    parser.add_argument("--config", type=str, default=None, required=True, help="Path to the config file")
    parser.add_argument("--log_time", type=str, default=None, required=True, help="Current time for logging purposes")
    args = parser.parse_args()

    # Load the config file
    with open(args.config, "r") as f:
        config = yaml.full_load(f)

    config['working_dir'] = os.path.join("./exp", config['data']['dataset'], config['name'], args.log_time)

    # Log config
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(config['working_dir']))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)
    
    config = DotMap(config)

    # Set the working directory
    Path(config.working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, config.working_dir)
    shutil.copy("main.py", config.working_dir)

    # Set the seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Load checkpoint if resuming
    if config.resume:
        global checkpoint
        checkpoint = torch.load(config.resume, weights_only=False)
        wandb_id = checkpoint['wandb_id']


    # Set wandb
    # wandb.require("core") # Use new backend for better performance
    wandb.init(project="SingleFrameActionAnticipation",
                name="{}_{}_{}".format(config.data.dataset, config.name, args.log_time) if not config.resume else None,
                id=wandb_id if config.resume else None,
                config=config, resume="allow")
    return config

if __name__ == "__main__":
    config = get_config()
    main(config)