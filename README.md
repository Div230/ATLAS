
ATLAS

Adaptive Token Clustering and Attention System — an experimental Transformer attention architecture exploring whether token clustering can reduce the effective attention search space while retaining useful contextual information.

ATLAS investigates a simple idea:

> Instead of requiring every query to interact directly with every token, can tokens be organized into clusters and represented at different levels of granularity?



The project evolved through several implementations and experimental variants. The repository contains the architecture, training code, experimental GPU kernels, validation code, and research artifacts. The large model checkpoints are hosted separately on Hugging at https://huggingface.co/Div230/atlas
The final run was implemented using the train_gpt.py and the tags are provided for both the 29M and 100M run. The class named ClusterSelfAttention16 is the final versiom used in the runs but throughout from 1-15 are experimental version being optimized overtime.

---

Model Artifacts

The trained model checkpoints and tensor artifacts are hosted in the Div230/atlas Hugging Face repository.

Run	Parameters	Training budget	Git tag	Artifacts

ATLAS 29M	~29M	1B tokens	atlas-29M-1B-final	Hugging Face
ATLAS 100M	~100M	1B tokens	atlas-100M-1B-final	Hugging Face


GitHub intentionally does not contain the large checkpoint files. The Git repository preserves the implementation and experiment metadata, while Hugging Face contains the large model and intermediate training artifacts.


---

1. Motivation

Standard causal self-attention allows a query at position t to attend to every preceding token.

For a sequence of length T, this produces an interaction space that grows quadratically:

O(T²)

This becomes increasingly expensive as context length grows.

ATLAS explores a hierarchical alternative.

Rather than treating every token as an equally granular attention target, the sequence is divided into clusters. A query can then obtain information through two levels:

The goal is to preserve detailed local information while providing a more compact representation of broader context.

This is an experimental architecture: the project investigates the trade-off between the computational savings of clustering and the information lost when multiple tokens are represented collectively.


---

2. ATLAS Architecture

ATLAS divides tokens into clusters.

For a cluster C_i containing L_i tokens, the implementation constructs a representative key/value representation from the tokens belonging to that cluster.

Conceptually:

Cluster C_i

The cluster representation provides a more compact way of accessing information from multiple tokens.

The architecture therefore has two conceptual attention paths:

Local token attention

A query can directly interact with tokens in its relevant local region.

This preserves fine-grained information.

Cluster-level attention

The query can also interact with cluster-level representations.

This provides access to broader information without requiring the query to directly evaluate every token represented by those clusters.

The intended trade-off is:
       reduced interaction space

The exact efficiency depends on the clustering strategy, cluster sizes, visibility rules, and implementation overhead.


---

3. Cluster Representations

A central component of ATLAS is the construction of cluster-level key/value representations.

Conceptually, for a cluster containing several token representations

The experimental implementations construct these representations from the Q/K states associated with the cluster.

The resulting representation can then participate in attention alongside the local token representations.

This is one of the central architectural ideas being investigated by ATLAS.


---

4. Experimental Triton Implementation

During the development of ATLAS, a custom Triton implementation was developed as an experimental performance trial.

Its purpose was to investigate whether one version of the clustered attention mechanism could be implemented efficiently at the GPU-kernel level.

The experimental implementation included custom GPU kernels for operations such as:

cluster centroid construction;

local attention;

centroid attention;

online softmax accumulation;

output accumulation; and

output normalization.


The experimental implementation is represented by files such as:

atlas_triton.py
atlas_triton_2.py

and associated Triton kernels.

Important distinction

The custom Triton implementation was experimental.

It was developed for a particular version of the ATLAS architecture as a performance/implementation trial.

It was not used to produce the final 29M or 100M training runs documented in this repository.

Therefore:

the Triton implementation should not be treated as the final ATLAS implementation;

the Triton performance characteristics should not be interpreted as the performance of the final 29M/100M runs;

and issues discovered specifically in that experimental implementation should not automatically be attributed to the final experiments.


The Triton work is preserved because it is part of the research and development history of ATLAS.


---

5. The Experimental Causal K-Leak

One of the most important issues discovered while experimenting with the Triton implementation was a causal K/V leakage problem caused by centroid construction.

The issue arose from the way a particular experimental version constructed cluster centroids.

The problematic operation was effectively:

idx = cluster_positions[b, c, :clen].long()

centroid_k[b, c] = (
    k[b, :, idx, :]
    .reshape(-1, D)
    .float()
    .mean(dim=0)
)

centroid_v[b, c] = (
    v[b, :, idx, :]
    .reshape(-1, D)
    .float()
    .mean(dim=0)
)

In other words, the centroid was constructed from the K/V representations of the entire cluster.

That creates a problem for causal language modeling.


---

Example

Consider:

Cluster C = [x₁, x₂, x₃, x₄, x₅]

Suppose the model is processing the query at x₂.

A causal model should only be able to use information from:

x₁, x₂

However, if the cluster representation is computed as:

K_c = mean(K₁, K₂, K₃, K₄, K₅)

V_c = mean(V₁, V₂, V₃, V₄, V₅)

then:

K_c / V_c

already contain information derived from:

x₃, x₄, x₅

The query at x₂ can therefore receive future information through the centroid.

Conceptually:

x₁
                 x₂ ◄── Query
                 x₃
                 x₄
                 x₅
                  │
                  ▼
          ┌────────────────┐
          │ Cluster centroid│
          └────────────────┘
                  │
                  ▼
             Query x₂

The local attention path may still be causal:

x₂ ─────► x₁, x₂       ✓

but the centroid path can leak:

x₂ ─────► centroid
              │
              ├── x₁
              ├── x₂
              ├── x₃  ← future
              ├── x₄  ← future
              └── x₅  ← future

This is the K/V leak encountered during the experimental Triton implementation.


---

6. Why the Leak Was Easy to Miss

The obvious token-level attention computation can still contain a correct causal mask.

For example, the experimental implementation included logic equivalent to:

causal = q_tok[:, None] >= k_tok[None, :]

This correctly prevents a query from directly attending to a future token.

However, that mask operates after the centroid has already been constructed.

That creates an important distinction:

Causal masking
                         │
                         ▼
Query ─────────────► individual K/V
                         ✓

versus:

Future tokens
     │
     ▼
centroid construction
     │
     ▼
future information already compressed
     │
     ▼
query attends to centroid
     ✗

Once future information has been incorporated into a single centroid, treating that centroid as one attention element cannot recover the original causal boundary.

This led to an important architectural lesson:

> Causal masking is insufficient when the representations being attended to were themselves constructed from future information.



Causality must therefore be considered during representation construction, not only during the final attention operation.


---

7. Causal Requirement for Cluster Representations

For an autoregressive model, a cluster representation visible to a query at position t must not contain information originating from positions greater than t.

The fundamental invariant is:

Information used to construct an attention representation
must not originate from a position later than the query.

Possible approaches to satisfying this requirement include:

prefix-dependent cluster representations;

causal cluster boundaries;

restricting centroid visibility to clusters entirely in the query's past;

constructing cluster summaries only at causal boundaries; or

maintaining multiple cluster representations corresponding to different causal prefixes.


The correct solution depends on the final ATLAS architecture and its desired computational properties.


---

8. Important Scope of the K-Leak

The K-leak described above belongs specifically to the experimental Triton implementation / ATLAS variant in which the issue was identified.

It should not be interpreted as saying that:

the final 29M run used the leaking Triton implementation;

the final 100M run used the leaking Triton implementation;

the reported final results were generated by that Triton implementation; or

the final experiments necessarily exhibit the same leakage.


The final 29M and 100M experiments were trained using the final ATLAS implementation, not the experimental custom Triton kernel described in this section.

The Triton implementation is documented because identifying this problem was part of the architectural development process.


---

9. Training Experiments

ATLAS was evaluated through progressively larger experimental runs.

29M / 1B

The first major experiment used an approximately:

29M parameters
1B training tokens

The run is preserved under:

atlas-29M-1B-final

Its artifacts include the ATLAS training checkpoints, dense baseline checkpoints, cluster metadata, and research metadata available from the experiment.

The complete model artifacts are hosted under:

Div230/atlas/29M-1B/


---

100M / 1B

The architecture was subsequently scaled to approximately:

100M parameters
1B training tokens

The run is preserved under:

atlas-100M-1B-final

The complete model artifacts are hosted under:

Div230/atlas/100M-1B/

The 100M experiment contains substantially more intermediate cluster metadata and checkpoints, reflecting the longer/larger experiment.


---

10. Dense Baseline

The experiments also preserve corresponding dense attention baselines.

The dense baseline provides a reference point for evaluating the effect of replacing the dense attention mechanism with ATLAS.

The comparison is important because reducing the number of attention interactions is not sufficient by itself.

The relevant questions are:

Does clustering reduce computation?

        AND

Does the resulting model retain comparable quality?

        AND

Does the overhead of clustering negate the theoretical savings?

The dense runs therefore provide an important control for interpreting the ATLAS experiments.


---

11. Research Trade-offs

ATLAS introduces several new sources of computation that do not exist in ordinary dense attention.

The potential benefits include:

fewer direct attention interactions;

compact representations of broader context;

potentially better scaling with sequence length;

opportunities for specialized GPU implementations.


But these benefits have to be weighed against:

cluster construction;

centroid computation;

cluster assignment;

centroid visibility management;

additional memory movement;

loss of information caused by summarization;

implementation complexity; and

maintaining strict causality.


Therefore, the real objective is not simply:

fewer attention scores

but rather:

lower overall computational cost
while maintaining useful information flow and model quality


---

12. Repository Organization

The GitHub repository contains the source implementation and research code.

Large model artifacts are hosted separately on Hugging Face.

ATLAS/
├── attention / ATLAS implementation
├── Triton experimental implementation
├── train_gpt.py
├── validation / testing code
├── data / tokenizer configuration
├── records/
└── experiment metadata

The model artifact repository is organized as:

Div230/atlas/
│
├── 29M-1B/
│   ├── atlas/
│   │   ├── atlas_metadata/
│   │   ├── checkpoints/
│   │   └── research/
│   │
│   └── dense/
│       └── checkpoints/
│
└── 100M-1B/
    ├── atlas/
    │   ├── atlas_metadata/
    │   ├── checkpoints/
    │   └── research/
    │
    └── dense/
        ├── checkpoints/
        └── research/

This separation keeps GitHub usable as a source repository while retaining the complete large-scale experiment artifacts on Hugging Face.


---

13. Research Artifacts

The Hugging Face repository preserves more than just final model weights.

For the ATLAS runs, intermediate metadata is retained at different training steps.

For example:

metadata_step_0
metadata_step_1000
metadata_step_2000
...
metadata_step_24415

These artifacts make it possible to investigate how the cluster structure changes during training rather than treating the final model as a black box.

The experiment repositories also contain training and research metadata where available.

This is useful for studying:

cluster evolution;

training dynamics;

attention structure;

checkpoint behavior;

ATLAS versus dense comparisons; and

the relationship between clustering and model performance.



---

14. Development History

ATLAS was developed iteratively rather than as a single finalized architecture.

The development process included:

Initial ATLAS concept
        │
        ▼
Clustered attention implementations
        │
        ▼
Experimental GPU/Triton implementation
        │
        ├── performance investigation
        │
        └── causal K/V leak discovered
        │
        ▼
Architecture / implementation refinement
        │
        ▼
Final ATLAS training implementation
        │
        ├── 29M / 1B experiment
        │
        └── 100M / 1B experiment

This distinction is important when interpreting the repository.

Not every implementation present in the codebase corresponds to the implementation used for the final reported runs. Some components exist specifically because they were useful experiments during the development of the architecture.


---

15. Current Status

ATLAS is an experimental research architecture, not a finished replacement for dense attention.

The project currently provides concrete experimental evidence and artifacts for investigating:

token clustering for attention;

hierarchical attention representations;

cluster-level K/V representations;

computational trade-offs;

scaling from approximately 29M to 100M parameters;

dense versus clustered attention;

GPU implementation strategies; and

causal constraints in hierarchical attention.


The most important architectural lesson uncovered during development is that compressing multiple tokens into an intermediate representation changes the causal information boundary.

A future version of ATLAS therefore needs to treat causal cluster construction as a first-class part of the architecture rather than relying solely on a causal mask at the final attention operation.


---

Model Artifacts

Hugging Face: Div230/atlas

ATLAS 29M — 1B tokens

ATLAS 100M — 1B tokens


Git tags:

atlas-29M-1B-final
atlas-100M-1B-final

The GitHub repository contains the code and research history; Hugging Face contains the large model artifacts.

Acknowledgment

The initial project structure and training scaffold were based on the OpenAI Parameter Golf repository, which was forked as the starting point for the project. The fork provided the underlying training setup, repository organization, and baseline infrastructure needed to begin experimentation. From that starting point, the project was substantially modified and extended to develop the ATLAS architecture, its clustering and attention mechanisms, experimental implementations, training configurations, and the subsequent 29M and 100M research runs.
