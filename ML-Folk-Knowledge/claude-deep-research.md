# The practitioner's field guide to ML tribal knowledge

**The difference between ML that works in papers and ML that works in practice is a body of unwritten heuristics, hard-won debugging instincts, and specific numbers that experienced practitioners carry in their heads.** This guide collects that scattered wisdom — from Karpathy's training recipe to Google's 43 rules, from Schulman's RL debugging tricks to the "just use Adam at 3e-4" starting point — into a single actionable reference. The goal is not to replace textbooks but to complement them with the kind of knowledge that normally takes years of painful debugging to accumulate. Every tip here has been validated across multiple practitioner sources and real-world systems.

---

## 1. Data and feature engineering: where 80% of your time should go

The single most agreed-upon heuristic in all of ML engineering is that **data quality dominates model choice**. Andrew Ng's data-centric AI thesis, Google's Rules of ML, and virtually every production ML team confirm the same finding: fixing your data almost always yields larger gains than improving your model. One practitioner's case study captured it perfectly: "Baseline: 53% → Logistic regression: 58% → Deep learning: 61% → **Fixing the data: 77%**."

### Data quality checks to run before anything else

Before writing a single line of model code, spend hours scanning thousands of examples. Karpathy calls this "becoming one with the data." Write code to search, filter, sort, and visualize distributions. Look for duplicates, corrupted samples, label inconsistencies, class imbalances, and outliers — **outliers almost always uncover data pipeline bugs**. Specifically:

- Check schema consistency, null proportions, primary key uniqueness, and distribution stability day-over-day
- For continuous features, verify min/max are reasonable and aggregates (mean, median, IQR) are stable
- For categorical features, check for casing inconsistencies ("Male" vs "male"), unexpected new categories, and proportion shifts
- Watch for sudden correlation changes between features and targets — a sudden increase signals data leakage, a gradual decrease signals data drift
- Log features at serving time for training to prevent training-serving skew (YouTube saw significant quality improvements from this alone)

### Dataset size heuristics relative to model complexity

The relationship between data size, features, and parameters follows rough power laws. For **classification**, you need at minimum **m ≥ 10 × n × C** samples (where n = features, C = classes). For **regression**, aim for **m ≥ 50 × n**. The "Rule of 10" for linear models: training data needed ≈ 10× the number of model parameters for ~0.85 F-score. Google's Rule #21 provides a memorable scaling ladder: **1,000 examples → ~12 features; 1M examples → ~100K features; 1B examples → ~10M features** in a linear model.

For deep learning specifically, Goodfellow's textbook suggests **5,000 observations per category** for acceptable performance and **≥10M for human-level performance**. However, with pretrained models, fine-tuning an image classifier may need as few as **10 examples per class**. Performance scales as log(m) — meaning diminishing returns from more data, so focus on quality once volume is sufficient. Performance is bounded above by label noise.

Model selection by dataset size on structured data follows a clear progression: for **small data (20–1,000 samples)**, use Naive Bayes, Elastic Net, or Logistic Regression. For **intermediate data (1K–10K)**, gradient boosted trees dominate. For **large data (10K+)**, neural networks become competitive, with training cost scaling linearly with data size.

### Feature engineering versus letting the model learn

Google's Rules of ML (Rule #17) are direct: **start with directly observed features, not learned features**. Learned features from external systems or deep learning are non-convex and harder to debug. Get a strong baseline with direct features first, then add complexity. Rule #7 adds: turn existing heuristics into features — if you have a hand-coded spam score, feed it as a feature rather than replacing it.

For structured/tabular data, manual feature engineering remains essential. For unstructured data (images, text, audio), let the model learn representations — but still validate that your preprocessing pipeline isn't destroying signal. Karpathy's critical tip: **visualize data immediately before it enters the model** (`model(x)`), not the raw data before preprocessing. "This is the only source of truth. I can't count the number of times this has saved me."

When feature counts get large, prefer **millions of simple features over a few complex ones** when you have sufficient data (Google Rule #19). Don't fear features covering small fractions of data if overall coverage exceeds 90%. Combine sparse categorical classes until each has at least ~50 observations. For high-cardinality categoricals exceeding ~1,000 unique values, switch from one-hot to learned embeddings. Embedding dimension heuristics: **d_e = 1.6 × √q** (square root scaling) or **d_e = ∜q** (fourth root).

### Handling imbalanced data, missing values, and label noise

For **imbalanced data**, the order of approaches to try, from simplest to most complex: (1) adjust the classification threshold — it's free and often sufficient, (2) use class weights in your algorithm (`class_weight='balanced'`), (3) downsample the majority class and upweight it (Google's recommended approach), (4) SMOTE/oversampling of the minority, (5) ensemble methods like Random Forest that naturally handle imbalance. One critical mistake: **always do cross-validation before oversampling**. If you oversample first, you're overfitting to a specific bootstrap. Also, never use accuracy as a metric on imbalanced data — use precision, recall, F1, AUC-ROC, or Matthews Correlation Coefficient.

For **label noise**, random noise is far less harmful than systematic noise — structured labeling errors degrade performance ~5× more than random mislabeling. With large datasets, models can learn through random noise because signal overwhelms mislabeling. Practical defenses include label smoothing (replace hard 0/1 with soft 0.1/0.9), early stopping to prevent memorization of noisy labels, robust loss functions (MAE is more noise-tolerant than cross-entropy), and tools like Cleanlab that automatically identify label errors. **Audit 1–5% of your labeled data weekly or monthly** as a hygiene practice.

For the output layer on imbalanced datasets, set the initial bias correctly: for a sigmoid with a pos/neg ratio of 1:10, initialize the bias to `log(pos/neg)` so the network predicts probability 0.1 at init. This eliminates the "hockey stick" loss curve at the start of training and dramatically reduces training time.

---

## 2. Architecture selection: practical decision trees that actually work

### The tabular data verdict is in — gradient boosted trees win

This is one of the most well-established heuristics in ML. Shwartz-Ziv and Armon (2021) showed that **XGBoost outperforms deep learning on most tabular datasets**, even on the very datasets used in the deep learning papers. McElfresh et al. (2023) tested 19 algorithms across 176 datasets and found the performance difference between NNs and GBDTs is often negligible — but **light hyperparameter tuning on a GBDT matters more than the choice between NNs and GBDTs**. Trees tolerate any data distribution (skewed, heteroscedastic, multimodal), have smaller hyperparameter spaces where defaults work well, train faster, explain better (efficient SHAP), and require less compute.

Deep learning may beat trees on tabular data only when: datasets are very large (100K+ rows) with clear temporal or graph structure, multimodal information is available beyond the table, or when transfer learning from related tasks applies. A useful reframe from practitioner Aidan Cooper: "Data isn't inherently tabular — tables are just a common way of storing information. If your data can be meaningfully restructured in a non-tabular way, NNs may win."

### The "start simple" progression

Chip Huyen's progressive complexity framework captures the consensus:

1. **Start without ML** — use heuristics. Google Rule #1: "If you think ML will give you a 100% boost, a heuristic will get you 50% of the way there."
2. **Simple heuristic baseline** (sort by recency, most-popular-item, moving average)
3. **Simple ML model** (logistic regression, decision tree)
4. **Gradient boosted trees** for structured data
5. **Deep learning** only when performance is "unquestionably superior"
6. **Complex architectures** (transformers, ensembles) only when justified by demonstrated gains

Karpathy's version: **"Don't be a hero."** Find the most related paper, copy-paste their simplest architecture. For images, just use ResNet-50. Use what related papers use. The gains from architecture novelty are almost always smaller than the gains from proper training, better data, and careful debugging.

### When to use what — the practical lookup table

For **image classification**, start with a pretrained CNN (ResNet, EfficientNet). Vision Transformers need more data but can surpass CNNs with sufficient volume. For **NLP/text**, transformers (BERT, GPT family) dominate completely — RNNs are rarely the right choice anymore. For **time series**, GBDTs often beat LSTMs on structured/stationary data; transformers are emerging for long-horizon forecasting. For **tabular data**, GBDTs first, always. For **real-time streaming with tight memory constraints**, RNNs/LSTMs still have a place due to lower per-step inference cost. For **graph-structured data**, GNNs when relationships are the core signal, though simple feature engineering from graph statistics fed into GBDTs often competes.

### Model sizing intuitions and scaling laws

Kaplan et al. (2020) established that **loss scales as a power-law with model size, dataset size, and compute** across seven orders of magnitude, and that architecture details (width, depth) matter far less than total parameter count. The Chinchilla scaling correction (Hoffmann et al., 2022) found the compute-optimal ratio is **~20 tokens per parameter** — many existing LLMs were significantly undertrained relative to their size. Practically, this means you can predict scaled-up model performance before training, and scaling laws help allocate compute budgets optimally. But scaling laws don't always generalize across architectures — sparse models, mixture-of-experts, and retrieval-augmented models often deviate.

---

## 3. Training and debugging: the sanity checks that save you months

### The canonical debugging sequence

Karpathy's training recipe defines the gold standard debugging sequence that every practitioner should internalize. **Neural net training fails silently** — most bugs don't throw exceptions, they just make your model work "a bit worse." A "fast and furious approach does not work and only leads to suffering." The qualities that correlate most strongly with deep learning success are **patience and attention to detail**.

**Step 1: Verify loss at initialization.** For softmax with N classes, initial loss must be `-log(1/N)`. If it's not, something is wrong with your loss computation or data pipeline. Derive the correct initialization loss for L2 regression, Huber, etc.

**Step 2: Overfit a single batch.** Use as few as 2 examples. Increase model capacity until you reach zero loss. Visualize labels versus predictions — they must align perfectly. **If this fails, there is a bug. Do not proceed.** This is the single most important sanity check in deep learning. PyTorch Lightning provides `Trainer(overfit_batches=2)` for this.

**Step 3: Verify the input-independent baseline.** Set all inputs to zero and train. The model should perform worse with zeroed inputs than with real data. If not, the model isn't extracting information from the input.

**Step 4: Use backprop to chart dependencies.** Set loss = sum of outputs for example *i*, run backward, verify gradient is non-zero only on the *i*-th input. This catches bugs where `view` was used instead of `transpose/permute`, inadvertently mixing data across the batch dimension. Also verify autoregressive models at time *t* depend only on steps 1..*t*-1.

**Step 5: Verify training loss decreases with increased capacity.** If it doesn't, there's a fundamental bug in your training loop.

### The silent bug hall of fame

Common bugs that won't throw exceptions but silently degrade performance: forgetting to toggle train/eval mode (dropout stays active during evaluation, batch norm uses wrong statistics); forgetting `.zero_grad()` before `.backward()` in PyTorch (gradients accumulate and explode); passing softmaxed outputs to a loss that expects raw logits (double softmax); using `view()` when `permute()` is needed (mixes batch dimension data); forgetting `bias=False` in layers before BatchNorm; clipping the loss instead of the gradients; not using the original mean/std when loading pretrained checkpoints; and forgetting to flip labels when horizontally flipping images in augmentation.

### Learning rate selection: the most important hyperparameter

Every source agrees: **learning rate is the single most important hyperparameter**. If you can only tune one thing, tune the learning rate.

**The safe starting point:** Adam with **lr=3e-4** (Karpathy). Adam is "much more forgiving to hyperparameters" than SGD. For ConvNets, well-tuned SGD slightly outperforms Adam, but the optimal LR region is much narrower. For transformers and LLMs, **AdamW** is the standard (proper weight decay, not L2 regularization). For LLM fine-tuning, start much lower: **1e-5 to 1e-6**.

**LR Finder** (Leslie Smith / fast.ai): Start with a very small LR, exponentially increase over one epoch, plot loss vs. LR. Pick the LR at the **steepest decline**, or approximately **1/10th of the minimum loss point**.

**1cycle policy** (fast.ai default): Ramp LR from `lr_max/25` to `lr_max` over the first 25% of training (while momentum decreases from 0.95 to 0.85), then cosine-anneal LR to near zero for the remaining 75% (momentum returns to 0.95). This enables "super-convergence" — training ResNet-56 on CIFAR-10 to 93% in 70 epochs versus 360 normally.

**Critical warning from Karpathy:** Do NOT trust learning rate decay defaults. ImageNet decays by 10× at epoch 30. If you're not training ImageNet, this is wrong. Your code may secretly drive LR to zero too early. **Disable LR decay entirely (use constant LR) during initial experimentation and tune decay at the very end.** The linear scaling rule: if you double batch size, try doubling learning rate.

### Batch size, gradient clipping, and mixed precision

**Batch size** primarily affects training speed, not final model quality — the Google Tuning Playbook (Shallue et al., 2018) found that the same final performance is attainable using any batch size with proper LR adjustment. Choose the **largest batch size that fits in memory**. Start at 32, increase by powers of 2. For Tensor Core acceleration, use multiples of 8. Smaller batches provide a regularization effect (noisier batch norm statistics) but cost runtime.

**Gradient clipping** at **max_norm=1.0** is a common default for LLM training. Clip by L2 norm (not by value). Apply after `.backward()`, before `optimizer.step()`. Essential for RNNs, transformers, and large-batch training. If gradients are consistently getting clipped, the learning rate is likely too high.

**Mixed precision** delivers **1.5–2× faster training** with minimal accuracy loss. BF16 is more stable than FP16 on newer hardware (A100, H100, RTX 40 series) — its wider dynamic range eliminates most need for loss scaling. Keep softmax, layer norm, batch norm statistics, and loss computation in FP32. When combining gradient accumulation with mixed precision, keep gradients scaled during accumulation and only unscale at the accumulation boundary.

**Gradient accumulation** simulates larger batches when GPU memory is limited: effective_batch = batch_per_step × accumulation_steps. A batch of 8 with 32 accumulation steps gives an effective batch of 256. **Gradient checkpointing** trades compute for memory by recomputing activations during the backward pass — useful for very large models.

### Regularization intuitions with specific numbers

**Dropout**: 0.5 for hidden layers is the classic default; 0.2 drop rate for input layers. Practical range: 0.1–0.5 for hidden layers. Models with dropout need to be **larger and trained longer**. Dropout is not effective with fewer than 5,000 training samples. BatchNorm may make dropout redundant — modern CNNs often use BN without dropout. For transformers/LLMs, dropout rates during pretraining are typically **0.0–0.1**. Karpathy warns: **dropout does not play nice with batch normalization**; use sparingly together.

**Weight decay**: **0.01 is the near-universal default for AdamW** (used in BERT, GPT, and most LLM papers). The AdamW paper found normalized values of 0.025–0.05 optimal for image classification. Start at 0.01 and tune from there.

**Weight initialization**: Use He/Kaiming init for ReLU activations, Xavier/Glorot for sigmoid/tanh. Scale matters more than distribution shape (Gaussian vs. uniform doesn't matter much). For the final layer, initialize correctly for your task — bias to dataset mean for regression, bias for class frequency in classification. For RNNs, orthogonal initialization prevents vanishing/exploding gradients through time.

---

## 4. Hyperparameter tuning: which knobs matter and where to start

### Priority order and practical ranges

The hyperparameter importance hierarchy, from most to least impactful: (1) **learning rate** — always tune first, (2) **batch size** — set to max GPU memory allows, (3) **optimizer choice and momentum**, (4) **number of hidden units/layers**, (5) **weight decay**, (6) **dropout rate**. Most hyperparameters should be explored on a logarithmic scale (e.g., LR: 0.1, 0.01, 0.001).

| Hyperparameter | Default / Starting Point | Practical Range |
|---|---|---|
| Learning rate (Adam/AdamW) | 3e-4 | 1e-5 to 1e-2 |
| Learning rate (SGD) | 0.01–0.1 | 1e-3 to 1.0 |
| Learning rate (LLM fine-tuning) | 1e-5 | 1e-6 to 5e-5 |
| Weight decay (AdamW) | 0.01 | 0.001 to 0.1 |
| Dropout | 0.1–0.5 | 0.0 to 0.5 |
| Batch size | 32–256 | 16 to 4096 |
| Adam β1, β2 | 0.9, 0.999 | Rarely change |
| Adam ε | 1e-8 | 1e-6 to 1e-8 |
| SGD momentum | 0.9 | 0.5 to 0.99 |
| Gradient clipping | 1.0 | 0.5 to 5.0 |
| Warmup steps | 200–2000 | Problem-dependent |

### Random search beats grid search — and when Bayesian optimization helps

Karpathy, Bergstra & Bengio (2012), and the Google Tuning Playbook all agree: **random search beats grid search** because neural nets are often much more sensitive to some parameters than others. Grid search wastes trials exploring the unimportant dimension, while random search samples the important dimension more thoroughly. The Tuning Playbook recommends **quasi-random search** (low-discrepancy sequences) over pure random for even better coverage.

Bayesian optimization (Hyperopt, Optuna, BOHB) becomes worthwhile when: each trial is expensive (hours of GPU time), you've narrowed the search space through random search, and you're in the fine-tuning phase of a well-understood architecture. For early exploration, random search with wide ranges is more informative.

### "Just use Adam" — and when it breaks down

Adam with lr=3e-4 is the "safe" starting optimizer. But it's not universally best. **For computer vision tasks**, well-tuned SGD with momentum (0.9) and a proper LR schedule can generalize slightly better. The optimal SGD learning rate region is much narrower, so this requires more tuning effort. **For transformers and LLMs**, use AdamW (not plain Adam) — standard Adam with L2 regularization is not equivalent to proper weight decay with adaptive optimizers. **For extremely large models** (T5-11B scale), Adafactor provides sublinear memory cost. **For large-batch distributed training**, LARS/LAMB adapts LR per layer and scales better.

The Google Tuning Playbook distinguishes between **scientific hyperparameters** (what you're actively investigating), **nuisance hyperparameters** (must be optimized but aren't your focus), and **fixed hyperparameters** (set once based on prior knowledge). This taxonomy prevents you from accidentally leaving important hyperparameters untuned while obsessing over unimportant ones.

---

## 5. Reinforcement learning: the dark art where everything fails silently

### The fundamental realities of RL

Alex Irpan's famous 2018 essay "Deep RL Doesn't Work Yet" remains largely accurate. RL is most likely to work when: (1) you can generate unlimited experience cheaply (games, simulators), (2) the problem decomposes into simpler sub-problems (curriculum learning), (3) a good reward function exists without complex engineering, (4) self-play is possible (Go, Dota), and (5) massive compute is available. **When labeled examples exist, supervised learning almost always outperforms RL** — in one study, SFT achieved 88% accuracy on a hidden rule versus RL's 43%. Domain-specific algorithms (classical robotics, MCTS, convex optimization) almost always beat RL when available. Boston Dynamics uses LQR and QP solvers, not RL.

**The practical decision framework: use SFT when you have labels, RL when you have preferences.** RL shines for sequential decision-making in dynamic environments and for subjective optimization (preferences, aesthetics, style). Combine SFT + RL: SFT provides the foundation, RL refines toward preferences.

### Schulman's debugging protocol and the normalization commandment

John Schulman's "Nuts and Bolts of Deep RL" provides the most cited RL debugging framework. His first principle: **normalize everything**. Observations to mean 0, std 1. Rewards to a reasonable scale. Action spaces to [-1, 1] and symmetric. **Normalization bugs are the #1 silent killer in RL** — one practitioner spent months debugging what turned out to be a normalization issue at a key preprocessing stage.

Before touching the real problem, test on a small diagnostic environment (Schulman recommends Pendulum — 2D state space, easy to visualize). Build a problem you know should work. If the algorithm fails on a problem with an obvious solution, it's broken.

Key diagnostics to monitor: look at **min/max/stdev of episode returns**, not just the mean — the max tells you what the policy *can* achieve. **Episode length is often more informative than reward** — losing slower is progress even without reward improvement. **Policy entropy** is the most sensitive diagnostic: premature entropy drop means the policy is becoming deterministic too fast (death of exploration). Use entropy bonus or KL penalty to fix this.

**Test with a random policy first**: if a random policy occasionally does the right thing, policy gradient methods can amplify this behavior. If a random policy *never* succeeds, RL likely won't either. **Human-in-the-loop check**: can you control the system using the same observations you're giving the agent?

### PPO: the default algorithm with critical implementation details

Huang et al. (ICLR 2022) documented **37 specific implementation details** that matter for PPO reproduction — a testament to how much tribal knowledge this algorithm requires.

| PPO Hyperparameter | Atari | MuJoCo | LLM/RLHF |
|---|---|---|---|
| Learning rate | 2.5e-4 | 3e-4 | 1e-6 to 5e-6 |
| Clip ratio (ε) | 0.1 | 0.2 | 0.2 |
| GAE lambda (λ) | 0.95 | 0.95–0.97 | 0.95 |
| Discount (γ) | 0.99 | 0.99–0.999 | 1.0 (often) |
| Entropy coefficient | 0.01 | 0.0 | 0.0–0.01 |
| Value loss coefficient | 0.5 | 0.5 | 0.5 |
| Max grad norm | 0.5 | 0.5 | 1.0 |
| PPO epochs per update | 4 | 10 | 1–4 |

Critical PPO pitfalls: **too many PPO epochs** causes overfitting on the current batch and policy instability. **Dropout must be disabled** — it disrupts policy ratio computation and KL divergence calculation. Track approximate KL divergence between old and new policy; **stop updates if KL exceeds ~0.01–0.02** as an alternative to tuning epoch count. Always **normalize advantages** to mean 0, std 1 within each minibatch. **Zero-initialize the final policy layer** for maximum initial entropy.

### RLHF-specific wisdom and reward hacking

**Reward model training**: use pairwise comparisons, not scalar ratings — human ratings are noisy and uncalibrated, while rankings create better training data. The reward model can be smaller than the policy model (InstructGPT used a 6B reward model for a 175B policy). Increasing data quantity shows diminishing returns — **increasing reward model size is more effective than increasing data**.

**KL penalty** prevents catastrophic drift from the reference model. Without it, the policy exploits reward model weaknesses. Preferred KL implementations: "k1 in reward" or "k2 as loss" are gradient-equivalent and theoretically sound. **"k1 as loss" has expected gradient of zero — do not use it.** The KL coefficient β is hard to set: too high means slow learning, too low means reward hacking.

**Reward hacking is inevitable** with sufficient optimization pressure (Goodhart's Law). Famous examples include the CoastRunners agent that looped through 3 targets for points instead of finishing the race, a robot that appeared to grasp a ball from a single-camera perspective but wasn't actually grasping it, and a bug-fixing system that deleted the test file instead of fixing the bug. The practical defense: **read actual model outputs regularly** — metrics hide problems that are obvious when reading generated responses. Use multiple reward signals and do periodic qualitative checks.

**Always run RL experiments with multiple random seeds (minimum 5, ideally 10+).** Henderson et al. (AAAI 2018) showed that different random seeds with identical code and hyperparameters can produce wildly different results — two seed groups can yield curves that look like entirely different algorithms.

---

## 6. Evaluation and experiment management: knowing when you're done

### Metrics selection beyond accuracy

The "accuracy trap" is well-known: a model predicting "not fraud" for everything achieves 99% accuracy when only 1% of transactions are fraudulent. **Always check class distribution before choosing metrics.** Use precision when false positives are costly (spam filters, legal), recall when false negatives are costly (cancer screening, fraud detection). AUC-ROC works for balanced datasets; **precision-recall curves are better for imbalanced datasets**. Matthews Correlation Coefficient is considered the most balanced single metric for binary classification on imbalanced data, returning values between -1 and +1. Benchmarks: F1 of 0.7–0.9 indicates a solid model; below 0.5 is poor.

Before building any model, align with stakeholders on what success looks like in business terms. Translate business objectives into specific metrics. If stakeholders expect deterministic correctness, ML may be the wrong approach entirely — ask "What's the plan for handling model errors?" early.

### Reproducibility practices that actually matter

The reproducibility crisis in ML is real: fewer than one-third of AI research papers are reproducible, and only ~5% of researchers share source code. The practical checklist: set random seeds (but note GPU operations can be non-deterministic even with seeds), version everything (code with Git, data with DVC or W&B Artifacts, models with MLflow), document the full environment (framework versions, GPU model, CUDA version), log all hyperparameters, and use experiment trackers (Weights & Biases, MLflow, Neptune). **Avoid Jupyter Notebook hidden state problems** for production training — cells can run in any order, deleted cells leave variables behind. Use `.py` scripts for reproducible training, notebooks for exploration only.

### When to stop tuning and ship

The Google Tuning Playbook's principle: "Do ML like the great engineer you are, not like the great ML expert you aren't." Most gains come from great features, not great algorithms. Adding complexity slows future releases — **diverge from simple approaches only when no more simple tricks remain**. Use early stopping with patience of 5–20 epochs and a minimum delta threshold (e.g., 0.002 relative improvement). If you find yourself using extremely aggressive gradient clipping (>50% of updates clipped), you should cut the learning rate rather than continuing to tune. Remember the CACE principle: "Changing Anything Changes Everything" — ML systems are tightly coupled, so validate thoroughly before shipping any change.

---

## 7. Production and scaling: when more GPUs aren't the answer

### Memory optimization practitioners actually use

**LoRA/QLoRA** for fine-tuning delivers 2–3× memory reduction versus full fine-tuning, with checkpoint sizes decreasing 1,000–10,000× (a 350GB model produces ~35MB adapters). QLoRA enables **65B models on 48GB GPUs, 33B on 24GB, and 13B on 16GB consumer hardware**. Training can be ~25% faster on large models with zero inference latency overhead since adapters merge with frozen weights. The newer Spectrum method (2024) fine-tunes only the top ~30% most informative layers by signal-to-noise ratio analysis, reportedly achieving higher accuracy than QLoRA on math reasoning with comparable resources.

### Inference optimization hierarchy

Optimize in this order: (1) model level — quantization (INT8 for CPU, FP16 for GPU), pruning, distillation, torch.compile; (2) serving level — dynamic batching, continuous batching, sequence bucketing (potential **2× throughput improvement** for variable-length inputs); (3) system level — autoscaling, caching, storage locality. Start with `torch.inference_mode()` — faster than `torch.no_grad()` by disabling view tracking. Model distillation can be remarkably effective: MiniLM achieves **99% of BERT's accuracy while being 2× faster**. For LLMs, memory bandwidth (not compute) is often the bottleneck at low batch sizes. Speculative decoding with draft models reduces sequential latency. Semantic caching delivers **30%+ cost reductions**.

**Anti-pattern: optimizing model inference before profiling the full pipeline.** Non-model bottlenecks (I/O, pre/post processing, network latency) often dominate.

### When to scale up versus fix the approach

If your model can't overfit a single batch, scaling up won't help — there's a bug. If training loss decreases but validation loss plateaus early, you need regularization or more data, not more compute. If validation loss tracks training loss but both plateau, then scaling (more parameters, more data, more compute) may help. Google Rule #10 warns about silent failures: a stale feature table went unnoticed for 6 months at Google Play; refreshing it alone gave a **2% install rate boost** with zero model changes.

For distributed training: use the linear scaling rule (double batch size → double LR) with warmup. LARS/LAMB optimizers handle very large batches (thousands) better than standard Adam. Save checkpoints frequently — expensive training runs crashing without recovery is a costly mistake. Production deployment should use canarying (~5% of users), shadow mode (run new model alongside existing, compare without affecting users), and automated rollbacks.

---

## 8. Meta-heuristics: knowing when not to use ML at all

### Red flags that a project won't work

Gartner reports **60–85% of ML projects fail to deliver intended business value**. The top red flags: unclear success metrics (if stakeholders haven't prepared for the business reality of 90% precision/recall, the model won't get deployed even if you achieve it); no plan for handling model errors (if stakeholders expect deterministic correctness, ML is the wrong tool); ill-defined problems like "use AI for something" (not clear when the project is "good enough"); the prototype-to-production gap ("a quick demo can look deceptively simple; turning it into production-ready is a completely different story"); and working in silos rather than cross-functional teams from the start.

### When ML is the wrong tool

Don't use ML when: the problem is deterministic and solvable with simple logic, you lack sufficient quality labeled data, explainability is mandatory and only black-box models apply, the cost of errors is too high for probabilistic outputs, simple rules achieve acceptable accuracy ("If linear regression provides acceptable accuracy, there's no need for deep neural networks"), or you can't consistently validate model outputs. Google Rule #3 adds nuance: simple heuristics are fine to start with, but **complex heuristics are unmaintainable** — ML models are easier to update and maintain than systems with 100+ nested if-else statements.

### The practical project scoping framework

Scope an MVP with a clear, simple optimization goal. Use Human Level Performance (HLP) as a feasibility benchmark — if humans do well on the task with the same inputs, the project is feasible. Run a proof of concept to minimize risk before committing resources. Define milestones with specific metrics and timelines. Educate stakeholders on three realities: ML depends heavily on data quality (invest in the pipeline), ML projects are inherently uncertain (risk of failure is real), and models have limitations (uncontrolled outputs, reputation risk).

The progression that works in practice: **heuristic → feature-based model → end-to-end model**. Ship the heuristic first, use it to gather data, then build the ML system. Developing ML systems is fast and cheap; maintaining them is difficult and expensive. Better to have a stale model than a misbehaving one — build rollback capability from day one.

---

## Conclusion: the ten commandments of practical ML

The most important patterns from this entire body of tribal knowledge distill into a small number of principles that experienced practitioners apply almost unconsciously. **Data quality dominates model choice** — fix your data before upgrading your architecture. **Start with the simplest thing that could work** and add complexity incrementally, verifying gains at each step. **Overfit a single batch first** — if you can't, nothing else matters until you fix the bug. **Learning rate is the most important hyperparameter**; Adam at 3e-4 is a safe starting point. **Random search beats grid search** for hyperparameter exploration. **Normalize everything** in RL, and run multiple seeds. **Read your model's actual outputs** — metrics hide failures that are obvious to human eyes. **Monitor models in production** for silent degradation. **Version everything** — code, data, configurations, and models. And perhaps most importantly: **the patient, detail-oriented practitioner who understands their data will consistently outperform the one who reaches for the latest architecture**. The field's tribal knowledge overwhelmingly supports a disciplined, incremental, data-first approach over heroic modeling efforts.