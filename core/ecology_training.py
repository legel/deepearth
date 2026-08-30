"""Fit ecological readers against frozen Earth4D state."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def environment_batch(
    model, source, index, *, factorized=True, habitat=False,
    neighbor_index=None, return_inputs=False,
):
    values, observed, coords, neighbors, manifolds, neighbor_values = \
        source.batch(index)
    if neighbor_index is not None:
        rows = neighbor_index[index]
        neighbors = source.coords[rows]
        manifolds = {"biological": source.phylo[source.cls[rows]]}
        neighbor_values = {
            "identity": source.cls[rows], "vision_dino": source.dino[rows]
        }
    context = model.context(coords, neighbors, manifolds, neighbor_values)
    present = {
        name: observed[name] if name in model.environment_names
        else torch.zeros_like(observed[name])
        for name in model.names
    }
    latent = model.encode(values, present, context)
    pooled = model._pool(latent, model.species_variable)
    logits = model._niche_species_logits(pooled, include_lens=False)
    if factorized:
        logits = model._factorized_environment_logits(pooled, logits)
    if habitat:
        logits = model._habitat_family_read(logits, values, coords, observed)
        logits = model._distribution_family_read(logits, values, coords)
    output = values[model.species_variable].long(), pooled, logits
    return (*output, values, observed, coords) if return_inputs else output


def freeze(model, prefixes):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = [
        parameter for name, parameter in model.named_parameters()
        if name.startswith(prefixes)
    ]
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def optimizer(parameters, design, device):
    algorithm = torch.optim.AdamW(
        parameters,
        lr=design.learning_rate,
        weight_decay=design.weight_decay,
        fused=device.startswith("cuda"),
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        algorithm, design.steps
    )
    return algorithm, schedule


def step_optimizer(loss, parameters, algorithm, schedule):
    algorithm.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, 5.0)
    algorithm.step()
    schedule.step()


def train_environment(model, source, design, device):
    parameters = freeze(model, (
        "environment_family_reader.", "environment_species_reader."
    ))
    modules = (
        model.environment_family_reader, model.environment_species_reader
    )
    model.eval()
    for module in modules:
        module.train()
    algorithm, schedule = optimizer(parameters, design, device)
    started = time.time()
    for step in range(1, design.steps + 1):
        index = source.train_index[torch.randint(
            len(source.train_index), (design.batch,), device=device
        )]
        with torch.no_grad():
            target, pooled, raw = environment_batch(
                model, source, index, factorized=False
            )
        logits = model._factorized_environment_logits(pooled, raw)
        _, family_mass = model._family_posterior(logits)
        target_family = model.species_family[target]
        loss = F.cross_entropy(logits, target) / math.log(logits.shape[-1])
        loss = loss + 0.5 * F.nll_loss(
            family_mass.clamp_min(1e-8).log(), target_family
        ) / math.log(max(model.family_count, 2))
        step_optimizer(loss, parameters, algorithm, schedule)
        if step == 1 or step % 100 == 0 or step == design.steps:
            rank = 1 + (logits > logits.gather(1, target[:, None])).sum(-1)
            family = model.species_family[logits.argmax(-1)].eq(target_family)
            print(
                f"step {step:>5}  environment_reader_loss {float(loss):.4f}  "
                f"top10 {float((rank <= 10).float().mean()):.3f}  "
                f"family {float(family.float().mean()):.3f}  "
                f"mrr {float(rank.float().reciprocal().mean()):.3f}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )


@torch.no_grad()
def evaluate_habitat(model, source, index):
    values, _, coords, _, _, _ = source.batch(index)
    target = model.species_family[values[model.species_variable].long()]
    prediction = model.habitat_family_expert(
        model._habitat_features(values, coords)
    ).argmax(-1)
    return prediction.eq(target).float().mean()


def train_habitat(model, source, design, device):
    parameters = freeze(model, "habitat_family_expert.")
    model.eval()
    model.habitat_family_expert.train()
    algorithm, schedule = optimizer(parameters, design, device)
    train = source.train_index
    cell = torch.floor(source.lat[train] / 0.5).long() * 10007 \
           + torch.floor(source.lon[train] / 0.5).long()
    cells = cell.unique()
    devices = [torch.cuda.current_device()] if cell.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20260828)
        held = cells[torch.randperm(len(cells), device=cell.device)][
            :max(1, len(cells) // 6)
        ]
    held_mask = torch.isin(cell, held)
    training_index = train[~held_mask]
    validation_index = train[held_mask]
    if len(validation_index) > 32_768:
        position = torch.linspace(
            0, len(validation_index) - 1, 32_768,
            device=validation_index.device,
        ).long()
        validation_index = validation_index[position]
    best_score, best_step, best_state = -math.inf, 0, None
    generator = torch.Generator(device=device).manual_seed(1337)
    started = time.time()
    for step in range(1, design.steps + 1):
        index = training_index[torch.randint(
            len(training_index), (design.batch,), device=device,
            generator=generator,
        )]
        values, _, coords, _, _, _ = source.batch(index)
        target = model.species_family[values[model.species_variable].long()]
        logits = model.habitat_family_expert(
            model._habitat_features(values, coords)
        )
        loss = F.cross_entropy(logits, target)
        step_optimizer(loss, parameters, algorithm, schedule)
        if step == 1 or step % 200 == 0 or step == design.steps:
            model.habitat_family_expert.eval()
            score = float(evaluate_habitat(model, source, validation_index))
            if score > best_score:
                best_score = score
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.habitat_family_expert.state_dict().items()
                }
            print(
                f"step {step:>5}  habitat_family_loss {float(loss):.4f}  "
                f"spatial_family {score:.6f}  best_step {best_step}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )
            model.habitat_family_expert.train()
    model.habitat_family_expert.load_state_dict(best_state)
    print(f"selected habitat family expert step {best_step}", flush=True)


def spatial_split(source):
    import numpy as np
    from scipy.spatial import cKDTree

    index = source.train_index
    cell = torch.floor(source.lat[index] / 0.5).long() * 10007 \
           + torch.floor(source.lon[index] / 0.5).long()
    cells = cell.unique()
    devices = [torch.cuda.current_device()] if cell.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(20260828)
        held = cells[torch.randperm(len(cells), device=cell.device)][
            :max(1, len(cells) // 6)
        ]
    calibration = torch.isin(cell, held)
    training_index = index[~calibration]
    calibration_index = index[calibration]
    training = training_index.cpu().numpy()
    held_rows = calibration_index.cpu().numpy()
    tree = cKDTree(source._frame(training))
    _, neighbor = tree.query(source._frame(training), k=source.n_neighbors + 4)
    candidate = training[neighbor]
    is_self = candidate == training[:, None]
    candidate = np.take_along_axis(
        candidate, np.argsort(is_self, axis=1, kind="stable"), axis=1
    )[:, :source.n_neighbors]
    _, neighbor = tree.query(source._frame(held_rows), k=source.n_neighbors)
    neighbor_index = source.neighbors.clone()
    neighbor_index[training_index] = torch.as_tensor(
        candidate, device=index.device
    )
    neighbor_index[calibration_index] = torch.as_tensor(
        training[neighbor], device=index.device
    )
    return training_index, calibration_index, neighbor_index


@torch.no_grad()
def evaluate_family(model, source, index, batch, neighbor_index, mixes):
    totals = torch.zeros(len(mixes), 3, device=index.device)
    model.eval()
    for start in range(0, len(index), batch):
        target, _, base, values, observed, coords = environment_batch(
            model, source, index[start:start + batch],
            neighbor_index=neighbor_index, return_inputs=True,
        )
        multimodal = model._habitat_family_multimodal_logits(values, observed)
        for position, mixture in enumerate(mixes):
            logits = model._habitat_family_read(
                base, values, coords, observed,
                multimodal_mix=mixture, multimodal_logits=multimodal,
            )
            logits = model._habitat_species_read(logits, values, coords)
            rank = 1 + (
                logits > logits.gather(1, target[:, None])
            ).sum(-1)
            totals[position] += torch.stack((
                (rank <= 10).float().sum(),
                model.species_family[logits.argmax(-1)].eq(
                    model.species_family[target]
                ).float().sum(),
                rank.float().reciprocal().sum(),
            ))
    return totals / len(index)


def train_family(model, source, design, device):
    prefixes = ("adapters.worldclim.", "habitat_family_multimodal_")
    parameters = freeze(model, prefixes)
    modules = (
        model.adapters["worldclim"],
        model.habitat_family_multimodal_norm,
        model.habitat_family_multimodal_reader,
        model.habitat_family_multimodal_head,
    )
    model.eval()
    for module in modules:
        module.train()
    algorithm, schedule = optimizer(parameters, design, device)
    training_index, calibration_index, neighbor_index = spatial_split(source)
    position = torch.linspace(
        0, len(calibration_index) - 1, min(len(calibration_index), 2_048),
        device=calibration_index.device,
    ).long()
    validation_index = calibration_index[position]
    mixes = torch.tensor(
        (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0), device=device
    )
    baseline = evaluate_family(
        model, source, validation_index, design.batch,
        neighbor_index, mixes[:1],
    )[0]
    best_score = float(baseline[2])
    best_step, best_mix = 0, 0.0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }
    print(
        f"family mesh baseline  B1 {float(baseline[0]):.6f}  "
        f"B6 {float(baseline[1]):.6f}  B23 {float(baseline[2]):.6f}",
        flush=True,
    )
    generator = torch.Generator(device=device).manual_seed(1337)
    started = time.time()
    for step in range(1, design.steps + 1):
        index = training_index[torch.randint(
            len(training_index), (design.batch,), device=device,
            generator=generator,
        )]
        values, observed, _, _, _, _ = source.batch(index)
        target = model.species_family[values[model.species_variable].long()]
        logits = model._habitat_family_multimodal_logits(values, observed)
        loss = F.cross_entropy(logits, target) \
               / math.log(max(model.family_count, 2))
        step_optimizer(loss, parameters, algorithm, schedule)
        if step == 1 or step % 250 == 0 or step == design.steps:
            scores = evaluate_family(
                model, source, validation_index, design.batch,
                neighbor_index, mixes,
            )
            current = None
            for mixture, metrics in zip(mixes, scores):
                eligible = float(metrics[0]) >= float(baseline[0]) - 0.001 \
                           and float(metrics[1]) >= float(baseline[1]) - 1e-6
                if eligible and (
                    current is None or float(metrics[2]) > float(current[1][2])
                ):
                    current = mixture, metrics
                if eligible and float(metrics[2]) > best_score:
                    best_score = float(metrics[2])
                    best_step = step
                    best_mix = float(mixture)
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                        if name.startswith(prefixes)
                    }
                    best_state["habitat_family_multimodal_mix"] = \
                        mixture.detach().cpu().clone()
            mixture, metrics = current or (mixes[0], scores[0])
            print(
                f"step {step:>5}  family_mesh_loss {float(loss):.4f}  "
                f"mix {float(mixture):.2f}  B1 {float(metrics[0]):.6f}  "
                f"B6 {float(metrics[1]):.6f}  B23 {float(metrics[2]):.6f}  "
                f"best_step {best_step}  best_mix {best_mix:.2f}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )
            for module in modules:
                module.train()
    state = model.state_dict()
    state.update({
        name: value.to(state[name].device) for name, value in best_state.items()
    })
    model.load_state_dict(state)
    print(
        f"selected multimodal family mesh step {best_step}  mix {best_mix:.2f}",
        flush=True,
    )


def distribution_targets(cache, source, device, scale="300m"):
    import numpy as np

    data = np.load(Path(cache) / "gbif_species_dist.npz")
    source_id = np.asarray(source.gbifID)
    order = np.argsort(data["gbifID"])
    position = np.searchsorted(data["gbifID"][order], source_id)
    bounded = np.minimum(position, len(order) - 1)
    hit = (position < len(order)) \
          & (data["gbifID"][order[bounded]] == source_id)
    row = np.full(len(source_id), -1, dtype=np.int64)
    row[hit] = order[bounded[hit]]
    return (
        torch.as_tensor(row, device=device),
        torch.as_tensor(data[f"idx_{scale}"], device=device).long(),
        torch.as_tensor(data[f"frq_{scale}"], device=device).float(),
    )


@torch.no_grad()
def evaluate_species(model, source, index, batch, neighbor_index, mixes):
    totals = torch.zeros(len(mixes), 3, device=index.device)
    model.eval()
    for start in range(0, len(index), batch):
        target, _, base, values, _, coords = environment_batch(
            model, source, index[start:start + batch], habitat=True,
            neighbor_index=neighbor_index, return_inputs=True,
        )
        expert = model._habitat_species_scores(values, coords)
        for position, mixture in enumerate(mixes):
            logits = model._habitat_species_read(
                base, values, coords, mixture, expert
            )
            rank = 1 + (
                logits > logits.gather(1, target[:, None])
            ).sum(-1)
            totals[position] += torch.stack((
                (rank <= 10).float().sum(),
                model.species_family[logits.argmax(-1)].eq(
                    model.species_family[target]
                ).float().sum(),
                rank.float().reciprocal().sum(),
            ))
    return totals / len(index)


def train_species(model, source, design, device):
    parameters = freeze(model, "habitat_species_")
    model.eval()
    model.habitat_species_query.train()
    algorithm, schedule = optimizer(parameters, design, device)
    training_index, calibration_index, neighbor_index = spatial_split(source)
    position = torch.linspace(
        0, len(calibration_index) - 1, min(len(calibration_index), 2_048),
        device=calibration_index.device,
    ).long()
    validation_index = calibration_index[position]
    mixes = torch.tensor((
        0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75,
        1.0, 1.25, 1.5, 2.0,
    ), device=device)
    baseline = evaluate_species(
        model, source, validation_index, design.batch,
        neighbor_index, mixes[:1],
    )[0]
    best_score = float(baseline[2])
    best_step, best_mix = 0, 0.0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("habitat_species_")
    }
    print(
        f"spatial validation baseline  B1 {float(baseline[0]):.6f}  "
        f"B6 {float(baseline[1]):.6f}  B23 {float(baseline[2]):.6f}",
        flush=True,
    )
    generator = torch.Generator(device=device).manual_seed(1337)
    target = source.cls[training_index]
    count = torch.bincount(
        target, minlength=source.n_classes
    ).clamp_min(1).float()
    sampled = torch.multinomial(
        count[target].rsqrt(), design.steps * design.batch,
        replacement=True, generator=generator,
    )
    started = time.time()
    for step in range(1, design.steps + 1):
        offset = (step - 1) * design.batch
        index = training_index[sampled[offset:offset + design.batch]]
        values, _, coords, _, _, _ = source.batch(index)
        target = values[model.species_variable].long()
        logits = model._habitat_species_scores(values, coords)
        target_family = model.species_family[target]
        family = model.species_family.unsqueeze(0) == target_family[:, None]
        loss = F.cross_entropy(logits.masked_fill(~family, -torch.inf), target) \
               / math.log(logits.shape[-1])
        step_optimizer(loss, parameters, algorithm, schedule)
        if step == 1 or step % 250 == 0 or step == design.steps:
            scores = evaluate_species(
                model, source, validation_index, design.batch,
                neighbor_index, mixes,
            )
            current = None
            for mixture, metrics in zip(mixes, scores):
                eligible = float(metrics[0]) >= float(baseline[0]) - 0.001 \
                           and float(metrics[1]) >= float(baseline[1]) - 1e-6
                if eligible and (
                    current is None or float(metrics[2]) > float(current[1][2])
                ):
                    current = mixture, metrics
                if eligible and float(metrics[2]) > best_score:
                    best_score = float(metrics[2])
                    best_step = step
                    best_mix = float(mixture)
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                        if name.startswith("habitat_species_")
                    }
                    best_state["habitat_species_mix"] = \
                        mixture.detach().cpu().clone()
            mixture, metrics = current or (mixes[0], scores[0])
            print(
                f"step {step:>5}  species_mesh_loss {float(loss):.4f}  "
                f"mix {float(mixture):.3f}  B1 {float(metrics[0]):.6f}  "
                f"B6 {float(metrics[1]):.6f}  B23 {float(metrics[2]):.6f}  "
                f"best_step {best_step}  best_mix {best_mix:.3f}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )
            model.habitat_species_query.train()
    state = model.state_dict()
    state.update({
        name: value.to(state[name].device) for name, value in best_state.items()
    })
    model.load_state_dict(state)
    print(
        f"selected species habitat mesh step {best_step}  mix {best_mix:.3f}",
        flush=True,
    )


@torch.no_grad()
def evaluate_distribution(
    model, source, index, batch, neighbor_index, family_mixes, species_mixes,
):
    totals = torch.zeros(
        len(family_mixes), len(species_mixes), 3, device=index.device
    )
    model.eval()
    for start in range(0, len(index), batch):
        target, _, base, values, _, coords = environment_batch(
            model, source, index[start:start + batch], habitat=True,
            neighbor_index=neighbor_index, return_inputs=True,
        )
        expert = model._distribution_scores(values, coords)
        for family_position, family_mix in enumerate(family_mixes):
            routed = model._distribution_family_read(
                base, values, coords, family_mix, expert
            )
            routed = model._habitat_species_read(routed, values, coords)
            for species_position, species_mix in enumerate(species_mixes):
                logits = model._distribution_species_read(
                    routed, values, coords, species_mix, expert
                )
                rank = 1 + (
                    logits > logits.gather(1, target[:, None])
                ).sum(-1)
                totals[family_position, species_position] += torch.stack((
                    (rank <= 10).float().sum(),
                    model.species_family[logits.argmax(-1)].eq(
                        model.species_family[target]
                    ).float().sum(),
                    rank.float().reciprocal().sum(),
                ))
    return totals / len(index)


def select_distribution(
    model, source, index, batch, neighbor_index, baseline=None
):
    family_mixes = torch.tensor(
        (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5), device=index.device
    )
    species_mixes = torch.tensor(
        (0.0, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5),
        device=index.device,
    )
    family_scores = evaluate_distribution(
        model, source, index, batch, neighbor_index,
        family_mixes, species_mixes[:1],
    )[:, 0]
    baseline = family_scores[0] if baseline is None else baseline
    family_best, family_mix = family_scores[0], 0.0
    for mixture, metrics in zip(family_mixes, family_scores):
        eligible = float(metrics[0]) >= float(baseline[0]) - 0.001 \
                   and float(metrics[1]) >= float(baseline[1]) - 1e-6
        if eligible and float(metrics[2]) > float(family_best[2]):
            family_best, family_mix = metrics, float(mixture)
    species_scores = evaluate_distribution(
        model, source, index, batch, neighbor_index,
        family_mixes.new_tensor([family_mix]), species_mixes,
    )[0]
    best, species_mix = species_scores[0], 0.0
    for mixture, metrics in zip(species_mixes, species_scores):
        eligible = float(metrics[0]) >= float(baseline[0]) - 0.001 \
                   and float(metrics[1]) >= float(baseline[1]) - 1e-6
        if eligible and float(metrics[2]) > float(best[2]):
            best, species_mix = metrics, float(mixture)
    return baseline, best, family_mix, species_mix


def train_distribution(model, source, cache, design, device):
    parameters = freeze(model, "distribution_")
    algorithm, schedule = optimizer(parameters, design, device)
    training_index, calibration_index, neighbor_index = spatial_split(source)
    row, target_index, target_frequency = distribution_targets(
        cache, source, device
    )
    training_index = training_index[row[training_index] >= 0]
    position = torch.linspace(
        0, len(calibration_index) - 1, min(len(calibration_index), 2_048),
        device=device,
    ).long()
    validation_index = calibration_index[position]
    baseline, _, _, _ = select_distribution(
        model, source, validation_index, design.batch, neighbor_index
    )
    best = baseline
    best_step, best_family_mix, best_species_mix = 0, 0.0, 0.0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name.startswith("distribution_")
    }
    generator = torch.Generator(device=device).manual_seed(20260907)
    sampled = torch.randint(
        len(training_index), (design.steps * design.batch,),
        device=device, generator=generator,
    )
    started = time.time()
    for step in range(1, design.steps + 1):
        offset = (step - 1) * design.batch
        index = training_index[sampled[offset:offset + design.batch]]
        values, _, coords, _, _, _ = source.batch(index)
        logits = model._distribution_scores(values, coords)
        selected_index = target_index[row[index]]
        frequency = target_frequency[row[index]]
        valid = selected_index >= 0
        frequency = frequency * valid
        frequency = frequency / frequency.sum(-1, keepdim=True).clamp_min(1e-8)
        loss = -(
            frequency * F.log_softmax(logits, -1).gather(
                1, selected_index.clamp_min(0)
            )
        ).sum(-1).mean() / math.log(logits.shape[-1])
        step_optimizer(loss, parameters, algorithm, schedule)
        if step % 500 == 0 or step == design.steps:
            _, current, family_mix, species_mix = select_distribution(
                model, source, validation_index, design.batch,
                neighbor_index, baseline,
            )
            if float(current[2]) > float(best[2]):
                best = current
                best_step = step
                best_family_mix = family_mix
                best_species_mix = species_mix
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                    if name.startswith("distribution_")
                }
                best_state["distribution_family_mix"] = \
                    current.new_tensor(family_mix).cpu()
                best_state["distribution_species_mix"] = \
                    current.new_tensor(species_mix).cpu()
            print(
                f"step {step:>5}  distribution_loss {float(loss):.4f}  "
                f"family_mix {family_mix:.3f}  species_mix {species_mix:.3f}  "
                f"B1 {float(current[0]):.6f}  B6 {float(current[1]):.6f}  "
                f"B23 {float(current[2]):.6f}  best_step {best_step}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )
    state = model.state_dict()
    state.update({
        name: value.to(state[name].device) for name, value in best_state.items()
    })
    model.load_state_dict(state)
    print(
        f"distribution baseline  B1 {float(baseline[0]):.6f}  "
        f"B6 {float(baseline[1]):.6f}  B23 {float(baseline[2]):.6f}",
        flush=True,
    )
    print(
        f"selected distribution step {best_step}  "
        f"family_mix {best_family_mix:.3f}  "
        f"species_mix {best_species_mix:.3f}  "
        f"B1 {float(best[0]):.6f}  B6 {float(best[1]):.6f}  "
        f"B23 {float(best[2]):.6f}",
        flush=True,
    )


@torch.no_grad()
def evaluate_community(model, source, index, batch, neighbor_index):
    species_mixes = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)
    family_mixes = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
    totals = torch.zeros(
        len(species_mixes), len(family_mixes), 3, device=index.device
    )
    model.eval()
    for start in range(0, len(index), batch):
        target, _, base, values, _, coords = environment_batch(
            model, source, index[start:start + batch], habitat=True,
            neighbor_index=neighbor_index, return_inputs=True,
        )
        distribution = model._distribution_scores(values, coords)
        base = model._habitat_species_read(base, values, coords)
        base = model._distribution_species_read(
            base, values, coords, expert_scores=distribution
        )
        base = model._distribution_tail_read(base, distribution)
        scale_scores = model._community_scale_scores(values, coords)
        species_expert = torch.logsumexp(torch.stack(scale_scores), 0) \
                         - math.log(len(scale_scores))
        family = torch.stack([
            model._scores_to_family(score).clamp_min(1e-8).log()
            for score in scale_scores
        ]).mean(0).exp()
        family = family / family.sum(-1, keepdim=True)
        for species_position, species_mix in enumerate(species_mixes):
            species_output = model._distribution_species_read(
                base, values, coords, mix=species_mix,
                expert_scores=species_expert,
            )
            for family_position, family_mix in enumerate(family_mixes):
                output = model._protected_family_tail(
                    species_output, family, family_mix
                )
                rank = 1 + (
                    output > output.gather(1, target[:, None])
                ).sum(-1)
                totals[species_position, family_position] += torch.stack((
                    (rank <= 10).float().sum(),
                    model.species_family[output.argmax(-1)].eq(
                        model.species_family[target]
                    ).float().sum(),
                    rank.float().reciprocal().sum(),
                ))
    metrics = totals / len(index)
    baseline = metrics[0, 0]
    eligible = (metrics[..., 0] >= baseline[0] - 0.001) \
               & (metrics[..., 1] >= baseline[1] - 1e-6)
    flat = metrics[..., 2].masked_fill(~eligible, -torch.inf).flatten().argmax()
    species_position = flat.div(len(family_mixes), rounding_mode="floor")
    family_position = flat.remainder(len(family_mixes))
    return (
        baseline,
        metrics[species_position, family_position],
        species_mixes[int(species_position)],
        family_mixes[int(family_position)],
    )


def train_community(model, source, cache, design, device):
    parameters = freeze(model, "community_scale_meshes.")
    algorithm, schedule = optimizer(parameters, design, device)
    training_index, calibration_index, neighbor_index = spatial_split(source)
    targets = {
        scale: distribution_targets(cache, source, device, scale)
        for scale in ("30m", "3km")
    }
    valid = torch.ones(len(training_index), dtype=torch.bool, device=device)
    for row, _, _ in targets.values():
        valid &= row[training_index] >= 0
    training_index = training_index[valid]
    if len(calibration_index) > 4_096:
        position = torch.linspace(
            0, len(calibration_index) - 1, 4_096, device=device
        ).long()
        calibration_index = calibration_index[position]
    baseline, _, _, _ = evaluate_community(
        model, source, calibration_index, design.batch, neighbor_index
    )
    best_score = float(baseline[2])
    best_step, best_species_mix, best_family_mix = 0, 0.0, 0.0
    best_metrics = baseline
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.community_scale_meshes.state_dict().items()
    }
    generator = torch.Generator(device=device).manual_seed(20260913)
    sampled = torch.randint(
        len(training_index), (design.steps * design.batch,),
        device=device, generator=generator,
    )
    started = time.time()
    for step in range(1, design.steps + 1):
        offset = (step - 1) * design.batch
        index = training_index[sampled[offset:offset + design.batch]]
        values, _, coords, _, _, _ = source.batch(index)
        features = model._habitat_features(values, coords)
        losses = []
        for scale, (row, target_index, target_frequency) in targets.items():
            logits = model.community_scale_meshes[scale](features)
            selected_index = target_index[row[index]]
            frequency = target_frequency[row[index]]
            valid_target = selected_index >= 0
            frequency = frequency * valid_target
            frequency = frequency / frequency.sum(
                -1, keepdim=True
            ).clamp_min(1e-8)
            losses.append(-(
                frequency * F.log_softmax(logits, -1).gather(
                    1, selected_index.clamp_min(0)
                )
            ).sum(-1).mean() / math.log(logits.shape[-1]))
        loss = torch.stack(losses).mean()
        step_optimizer(loss, parameters, algorithm, schedule)
        if step == 1 or step % 500 == 0 or step == design.steps:
            _, current, species_mix, family_mix = evaluate_community(
                model, source, calibration_index, design.batch, neighbor_index
            )
            if float(current[2]) > best_score:
                best_score = float(current[2])
                best_step = step
                best_species_mix = species_mix
                best_family_mix = family_mix
                best_metrics = current
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.community_scale_meshes.state_dict().items()
                }
            print(
                f"step {step:>5}  community_scale_loss {float(loss):.4f}  "
                f"species_mix {species_mix:.3f}  family_mix {family_mix:.3f}  "
                f"B1 {float(current[0]):.6f}  B6 {float(current[1]):.6f}  "
                f"B23 {float(current[2]):.6f}  best_step {best_step}  "
                f"elapsed {time.time() - started:.1f}s",
                flush=True,
            )
            model.train()
    model.community_scale_meshes.load_state_dict(best_state)
    model.community_scale_species_mix.fill_(best_species_mix)
    model.community_scale_family_mix.fill_(best_family_mix)
    print(
        f"community scale baseline  B1 {float(baseline[0]):.6f}  "
        f"B6 {float(baseline[1]):.6f}  B23 {float(baseline[2]):.6f}",
        flush=True,
    )
    print(
        f"selected community scale step {best_step}  "
        f"species_mix {best_species_mix:.3f}  "
        f"family_mix {best_family_mix:.3f}  "
        f"B1 {float(best_metrics[0]):.6f}  B6 {float(best_metrics[1]):.6f}  "
        f"B23 {float(best_metrics[2]):.6f}",
        flush=True,
    )


def train_ecological_readers(model, source, cache, design, device):
    stages = (
        ("habitat family", train_habitat, design.habitat),
        ("multimodal family mesh", train_family, design.family),
        ("environment reader", train_environment, design.environment),
        ("species habitat mesh", train_species, design.species),
    )
    for name, function, stage in stages:
        print(f"{name} phase {stage.steps} steps", flush=True)
        function(model, source, stage, device)
    print(f"distribution mesh phase {design.distribution.steps} steps", flush=True)
    train_distribution(model, source, cache, design.distribution, device)
    print(
        f"multiscale community mesh phase {design.community.steps} steps",
        flush=True,
    )
    train_community(model, source, cache, design.community, device)
