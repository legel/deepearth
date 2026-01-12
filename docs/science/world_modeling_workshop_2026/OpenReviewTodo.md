# Open Review To Do Items

As background, this paper is in preparation for the [2026 World Modeling Workshop](https://world-model-mila.github.io/) at the Mila - Quebec AI Institute. In this directory, please find the LaTeX directory for generating the paper.

Paper Status: *accepted*

## Addressing Reviewer Feedback

1. Introduction and context of the approach
    - [ ] Revise introduction for world modeling audience (vs ecology audience) (Reviewer [VajQ])
    - [ ] Clarify distinction between world model and neural field in introduction (Reviewer [QJfa])
    - [ ] Expand literature review in introduction (Reviewers [QS1P], [QJfa])
    - [ ] Discuss dependence on coordinates and species labels alone and its limitations (Reviewer [QS1P])
    - [ ] More clarity on multimodality (Reviewer [QJfa])
2. Datasets beyond Live Fuel Moisture Content
    - [ ] Consider datasets beyond Live Fuel Moisture Content (Reviewers [MrdT] and [QS1P])
3. Context of the results
    - [ ] More discussion of the results to date to contextualize the impact and opportunity (Reviewer [VajQ])
    - [ ] Need a more direct conclusion (Reviewer [VajQ])
4. Minor
    - [ ] Fix GitHub link issue (Reviewer [VajQ])
5. If time permits
    - [ ] Learned hash encodings (Reviewer [QS1P])
    - [ ] Longer temporal horizons (Reviewer [QS1P])
    - [ ] Cross-region generalization (Reviewer [QS1P])
    - [ ] Supplemental material (e.g., ablations) (Reviewer [VajQ])

## Per Reviewer Breakdown

### Reviewer MrdT

1. Recommendation to consider a dataset beyond Live Fuel Moisture Content.

### Reviewer QS1P

1. Needs a more comprehensive introduction, especially in terms of literature review.
2. ``Dependence on coordinates and species labels alone may limit performance on tasks requiring rich environmental cues."  --> Needs a sentence or so to discuss this
3. See MrDT above
4. Learned hash encodings if time permits
5. Longer temporal horizons or cross-region generalization

### Reviewer QJfa

1. More clarity on multimodality
2. See (1) in Reviewer QS1P
3. Distinction of world model versus a neural field

### VajQ

1. Introduction clarity to a world modelling audience vs ecology audience
2. Contextualizing performance
3. GitHub link error
4. Conclusion and impact
5. Supplementals (e.g., ablations)
