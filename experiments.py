import os

import torch
from tqdm.notebook import tqdm
from utils.metrics import (
    aggregate_metrics,
    get_classification_scores,
    init_experiment_metrics,
    log_metrics,
)
from utils.train import (
    create_data_loaders,
    evaluate,
    train_one_epoch,
    validate,
)


def run_experiment(
    model,
    config,
    save_experiment=True,
    validate_experiment=True,
    evaluate_experiment=True,
    display_logs=True,
):
    """
    Runs the full training and evaluation pipeline.

    Args:
        model (torch.nn.Module): The model to train and evaluate.
        config (dict): Configuration dictionary containing dataset and training hyperparameters.
        validate (bool): wether to evaluate model on validaion set
        evaluate (bool): wether to evaluate model on test set
    Returns:
        dict: Aggregated metrics for all phases
        dict: Scores on the test set
    """
    # Initialize experiment metrics dictionnary
    metrics = init_experiment_metrics(include_val_metrics=validate_experiment)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config=config,
        include_val_loader=validate_experiment,
        include_test_loader=evaluate_experiment,
    )

    # Initialize
    device = config["device"]
    num_epochs = config["num_epochs"]
    criterion = config["criterion"]()
    optimizer = config["optimizer"](model.parameters())
    output_size = config["output_size"]

    model.to(device)

    for epoch in tqdm(range(num_epochs), desc="Experiment Progress", unit="epoch"):
        # Train phase
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        aggregate_metrics(metrics, train_metrics, phase="train")

        if val_loader is None:
            if display_logs:
                log_metrics(epoch, num_epochs, metrics["train"])
            continue

        # Validation phase
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )
        aggregate_metrics(metrics, val_metrics, phase="val")
        if display_logs:
            log_metrics(epoch, num_epochs, metrics["train"], metrics["val"])

    if save_experiment:
        save_model(model, config)

    # Evaluation phase
    if test_loader:
        test_metrics = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            output_size=output_size,
        )
        if output_size > 1:
            get_classification_scores(
                test_metrics,
                config["scores"],
            )
        metrics["test"] = test_metrics

    return metrics


def save_model(model, config):
    model_name = model.__class__.__name__
    save_path = os.path.join(config["results_dir"], f"{model_name}.pt")
    torch.save(model.state_dict(), save_path)
