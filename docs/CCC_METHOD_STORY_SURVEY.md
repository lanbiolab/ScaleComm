# 高水平细胞通讯方法论文叙事调研

本调研面向 SpatialSelf 的重构叙事。范围按 Nature Communications、Genome Biology、Nature Methods、Nature Biotechnology、Nature Genetics、Nature Computational Science 等同级或更高水平方法论文来取样；重点不是罗列所有工具，而是总结这些文章如何把一个 CCC 方法讲成一篇可发表的方法学论文。

## 1. 高水平方法论文的主流叙事类型

### A. 数据库 + 统计推断型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| CellChat | Nature Communications 2021 | 从单细胞表达和 curated LR database 出发，构建系统级通信网络，强调 pathway-level、cell-group-level、network pattern 和可视化解释。 |
| ICELLNET | Nature Communications 2021 | 强调 curated LR database、multi-subunit / molecular family 分类和可解释 score，目标是让生物学家能 dissect intercellular communication。 |
| CellPhoneDB | Nature Protocols / later versions | 叙事重点是可靠 LR database、multi-subunit ligand/receptor complex、permutation/statistical scoring，工具生态强。 |

这些文章的故事公式：

> 现有 scRNA-seq 能看到细胞类型，但不知道细胞之间怎么交流  
> -> ligand-receptor knowledge 可以提供可解释假设  
> -> 需要系统化、可视化、可统计检验的工具  
> -> 通过真实数据案例证明能恢复已知 biology，并发现可解释 interactions。

它们的强项：

- database 可信；
- 输出容易解释；
- pathway / family / cell-type network 层面好讲；
- 真实数据故事强。

它们的弱点：

- 空间约束弱或后加；
- 容易把高表达 LR 排前；
- 弱信号和空间 niche-specific signal 不一定有优势；
- ground truth benchmark 往往不充分。

对 SpatialSelf 的启发：

SpatialSelf 不应该和这类方法拼“数据库更全”。我们的叙事应该说：数据库型方法解决了 LR hypothesis generation，但没有充分解决 spatially constrained weak signal 和 false-positive calibration。

### B. 下游 target / 机制链型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| NicheNet | Nature Methods 2020 | 不只看 ligand 和 receptor 共表达，而是把 ligand activity 和 receiver target genes 连接起来，用 downstream transcriptional response 提高生物解释力。 |
| SpaTalk | Nature Communications 2022 | 在空间数据中整合 ligand-receptor-target knowledge graph，强调 LRT chain 而不是单纯 LR coexpression。 |

故事公式：

> 只看 ligand-receptor 共表达不够，因为通信真正产生功能结果的是 receiver response  
> -> 需要把 upstream ligand 和 downstream target response 连起来  
> -> 通过 known pathways / target genes / tissue case 验证。

强项：

- 生物逻辑更完整；
- 更容易回应“这像不像真的通信”；
- 真实数据案例有说服力。

弱点：

- target database 不完整；
- 受 pathway prior 强约束；
- 对弱表达、空间多尺度、FDR 控制不一定强；
- 不一定适合所有 tissue 和 all LR。

对 SpatialSelf 的启发：

我们必须补强 downstream validation。现在 Phase 6 的 CD48-CD244 是 receptor-expression proxy，不够强。后续应做 NicheNet/OmniPath/Reactome/CellChatDB target signature 验证，让“像不像合理通信”不只靠 SpatialSelf score。

### C. 空间 optimal transport / 距离约束型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| SpaOTsc | Nature Communications 2020 | 用 optimal transport 从 scRNA-seq 恢复空间关系，并推断 spatial signaling relationships。 |
| COMMOT | Nature Methods 2023 | 用 collective optimal transport 显式建模空间距离、LR competition 和不同 ligand/receptor species 的通信分配。 |

故事公式：

> CCC 发生在空间中，scRNA-seq 的非空间 LR 推断会有大量假阳性  
> -> 空间距离和 molecule competition 是关键约束  
> -> optimal transport 提供 principled spatial communication allocation  
> -> 在模拟和真实 ST 中展示 spatial communication map 和 downstream analysis。

强项：

- 空间理论清楚；
- distance/range constraint 合理；
- 适合 paracrine-like signal；
- 方法学形式漂亮。

弱点：

- 通常仍强依赖表达和距离；
- LR identity 的机制差异不一定充分建模；
- 对 contact、short-range、long-range niche-restricted signal 的统一处理需要额外设计；
- 弱信号容易被高表达竞争掉。

对 SpatialSelf 的启发：

我们要承认 COMMOT 的空间 OT 是强 baseline。SpatialSelf 的差异不能写成“我们也用了空间距离”，而要写成：

> SpatialSelf learns LR-specific spatial activity patterns beyond a fixed expression-distance allocation, enabling weak/context-specific CCC recovery with FDR control.

### D. 知识图谱 / LRT 网络型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| SpaTalk | Nature Communications 2022 | 用 graph network 和 knowledge graph 整合 LR proximity 与 LRT co-expression。 |
| CytoTalk | Nature Communications 2021 | De novo 构建 signal transduction networks，从 cell type pair 的表达关系中推断 communication program。 |

故事公式：

> 单个 LR pair 太碎片化，真正的通信是网络和通路  
> -> 用 graph / knowledge graph / de novo network 把 LR、target、pathway 串起来  
> -> 证明得到的网络模块能解释 tissue/cell-state biology。

强项：

- 叙事更像 biology；
- 可以讲 module、pathway、program；
- 对审稿人更容易形成“不是打分器”的印象。

弱点：

- prior 或 network construction 复杂；
- benchmark 难；
- 对假阳性控制和弱信号 recovery 仍需独立证明。

对 SpatialSelf 的启发：

我们目前还缺 CCC module 层面的结果。后面应该把 LR score profile 聚类成 communication modules，例如 T-zone chemokine module、GC module、melanoma immune-suppression module。这样文章会更像成熟方法，而不是 top LR 排名展示。

### E. 深度图模型 / 空间邻域表达型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| DeepLinc | Genome Biology 2022 | 用 variational graph autoencoder 从 single-cell spatial transcriptomics 重构 cell interaction landscape，强调 de novo 和 latent interaction representation。 |
| DeepTalk | Nature Communications 2024 | 用 subgraph-based graph attention network 在 single-cell resolution ST 中推断 CCC。 |
| CellNEST | Nature Methods 2025 | 用 attention mechanisms 从 ST 中识别 cell-cell relay networks，强调 relay、attention 和 spatial graph。 |
| NicheCompass | Nature Genetics 2025 | 用 graph deep learning 建模 cellular communication，学习可解释 niche embedding，并展示 atlas-scale scalability。 |
| Niche-DE / niche-LR | Genome Biology 2024 | 先识别 niche-associated genes，再结合 ligand-target prior 解释 context-dependent cell-cell interactions。 |

故事公式：

> 高分辨率 ST 让细胞邻域变复杂，传统 pairwise LR scoring 不够  
> -> 图模型可以建模 local spatial context / subgraph / relay / niche  
> -> deep representation 可以捕捉非线性 communication pattern  
> -> 通过 benchmark、真实组织案例、解释性模块和 scalability 证明有用。

强项：

- 适合高分辨率数据；
- 可以讲 neighborhood、relay、niche；
- 模型新意强；
- 容易做漂亮空间图。

弱点：

- 容易被审稿人问：是不是黑箱？是不是模型容量？是不是过拟合？
- benchmark 和 ablation 必须很强；
- FDR/p-value 通常不是自然输出；
- 如果只展示案例，会被认为证据弱。

对 SpatialSelf 的启发：

SpatialSelf 现在已经朝这个方向走：graph wavelet + LR identity + self-supervised + FDR。我们必须把深度模型的常见质疑提前堵住：

- 和 CellNEST / DeepTalk 比；
- 做 no LR identity、no wavelet、GAT、MLP、pseudo-label ablation；
- 做 permutation/FDR；
- 做 realistic benchmark；
- 做 source data 和 reproducibility。

### F. 多样本 / context-dependent communication 型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| Tensor-cell2cell | Nature Communications 2022 | 用 tensor decomposition 同时建模 cell-cell、LR、context，多样本中发现 context-driven CCC programs。 |
| CrossChat | Nature Communications 2024 | 从 global/local hierarchical structure 角度分析 CCC。 |
| MultiNicheNet | bioRxiv / widely used package | 强调 differential CCC，在多样本、多条件下做稳健 communication inference。当前不应写成 NC/GB 以上正式方法文，但叙事方向值得参考。 |

故事公式：

> 单样本 top LR 不够，疾病/发育/空间 context 会改变通信 program  
> -> 需要跨 context 建模通信变化  
> -> 通过 multi-condition data 找到 disease/state-specific communication programs。

强项：

- 更接近真实研究问题；
- 多样本统计更强；
- 容易讲 disease biology；
- 比单样本案例更稳。

弱点：

- 对数据要求高；
- 空间分辨率和平台差异更复杂；
- 方法实现和评价更难。

对 SpatialSelf 的启发：

如果要冲更高层次，SpatialSelf 后期最好增加跨样本 / cross-tissue 的 CCC profile 或 module analysis。现在我们有 lymph node、tonsil、melanoma 三类背景，但还没有真正讲成 multi-context communication program。

### G. niche response / spatial ligand-target activity 型

代表方法：

| 方法 | 期刊 | 核心叙事 |
|---|---:|---|
| Niche-DE / niche-LR | Genome Biology 2024 | 不是直接预测 LR edge，而是先识别 cell-type-specific niche-associated genes，再用 ligand-target prior 解释 niche-driven response。 |
| Renoir | Nature Communications 2026 | 在空间 niche 中刻画 ligand-target activity，把 ligand、receptor、target、cell-type composition 和 pathway activity 连接起来。 |
| FineST | Nature Communications 2026 | 用 histology foundation / contrastive learning 提升高分辨率 ST 表达与 LR 分析，强调图像 + ST 融合后的 fine-grained CCC pattern。 |

故事公式：

> 真正重要的不是某个 LR 是否共表达，而是它是否在特定空间 niche 中驱动 receiver 状态或 target activity  
> -> 先建模 niche / histology / downstream response  
> -> 再把 LR 或 ligand-target activity 放回空间结构中解释。

强项：

- 很符合审稿人对“functional validation”的期待；
- 能避免只看 LR top list；
- 和高分辨率 ST / histology 结合后故事更前沿。

弱点：

- 对 target prior、cell type annotation、histology quality 依赖强；
- ground truth 更难；
- 不一定直接输出 pair-level CCC benchmark 指标。

对 SpatialSelf 的启发：

这是我们下一阶段最该吸收的叙事：SpatialSelf 不能只说“我找到 LR pair”，要进一步证明 top LR 的空间 activity 与 receiver target state / niche response 一致。Phase 6 的 receptor-expression proxy 只是最低限度，后面应升级成 ligand-target / pathway target signature。

## 2. 这些高水平论文共同的“审稿人认可证据链”

高水平 CCC 方法论文通常不是靠一个指标打动审稿人，而是靠下面这条链：

1. **明确缺口**
   不是“我们又提出一个 CCC 工具”，而是一个清楚的痛点：
   - scRNA-seq 没空间；
   - 空间方法只看距离/共表达；
   - LR pair 没有 downstream response；
   - deep model 黑箱；
   - 多样本 context 变化没建模；
   - high-expression false positives 很多。

2. **方法设计正好对应缺口**
   每个模块都要有理由：
   - LR database -> biological prior；
   - target genes -> functional response；
   - OT -> spatial allocation；
   - GNN -> spatial neighborhood/subgraph；
   - tensor -> context；
   - permutation -> FDR。

3. **benchmark 不偏向自己**
   越好的文章越会证明：
   - simulation 不只是表达高 + 距离近；
   - 使用真实坐标/真实表达背景；
   - 和主流 baseline 比；
   - 做 robustness；
   - 说明失败 baseline 和不可运行方法。

4. **ablation 证明不是模型容量**
   深度方法尤其需要：
   - no graph；
   - no LR identity；
   - no distance；
   - no expression；
   - GAT/GCN/MLP 替代；
   - pseudo-label 策略。

5. **真实数据不是只挑案例**
   需要先全局统计，再展示案例：
   - positive/negative controls；
   - region enrichment；
   - downstream target correlation；
   - pathway enrichment；
   - cell-type specificity；
   - FDR。

6. **可解释性**
   高水平文章通常会给：
   - pathway / family / module；
   - LR embedding 或 interaction module；
   - niche program；
   - network hierarchy；
   - representative spatial map。

7. **可复现性和可扩展性**
   特别是高分辨率 ST：
   - runtime；
   - memory；
   - sample scale；
   - tutorial；
   - code/data availability。

## 3. SpatialSelf 最适合的叙事定位

我们不要把 SpatialSelf 讲成“又一个 GNN CCC 方法”。更强的叙事应该是：

> SpatialSelf is a self-supervised, LR-identity-aware spatial communication framework that learns LR-specific multiscale spatial activity and provides FDR-calibrated weak/context-specific CCC discovery.

中文大白话：

> 传统方法容易把“表达高、距离近”的 LR 排前；深度图方法能看邻域但容易黑箱、缺少 FDR。SpatialSelf 的核心卖点是：把 LR identity、空间多尺度图结构和 self-supervised scoring 结合起来，专门解决弱信号和高表达假阳性之间的 tradeoff，并用 permutation/FDR 给统计边界。

这个叙事里，SpatialSelf 的两个主亮点应该是：

1. **弱信号 / 假阳性 tradeoff**
   - 不说 SpatialSelf 每个指标都最好；
   - 说它在 realistic benchmark 中更适合筛 weak/context-specific CCC，同时压制 high-expression false positives；
   - 指标必须一起报：weak recall + empirical FDR + top-k precision + high-expression FP rate。

2. **LR identity-aware multiscale spatial modeling**
   - 不同 LR 的空间行为不同；
   - contact、paracrine、chemokine-like niche 不能用同一个固定 KNN / distance rule 解释；
   - graph wavelet 给多尺度空间表示；
   - LR identity embedding 让模型学习 LR-specific activity pattern。

第三个辅助亮点：

3. **FDR-calibrated discovery**
   - 这是很多 deep CCC 方法不强调的；
   - 我们可以把它作为 reviewer-facing 可信度来源；
   - 但要保守：当前最强是 LR-pair / cell-type-pair group-level，不要吹 edge-level。

## 4. 我们当前证据能支撑什么

根据当前 repo 的 claim boundary 和 Phase 6 结果，已经能支撑：

1. **realistic benchmark 方向**
   - controlled simulation 已经覆盖 contact、short/medium paracrine、long-range niche、weak signal、high-expression false positive；
   - real-background benchmark 有 lymph node、tonsil、melanoma；
   - 这能回应“simulation too artificial / too aligned with method”。

2. **baseline comparison**
   - retained baselines 包括 COMMOT、CellChat、SpaTalk、CellNEST；
   - LR-Coexp 已去掉；
   - DeepTalk / NICHES 目前仍更像 status methods，不能写成完整 formal comparison。

3. **weak-signal / FP tradeoff**
   - 现在可以写成 metric-specific tradeoff；
   - 不能写成所有指标、所有组织、所有 baseline 全面第一。

4. **ablation**
   - 已经有 LR identity、graph wavelet、distance、expression、MLP/GAT/pseudo-label 等方向；
   - 但如果投高水平，最好进一步做 longer-budget / more replicates 或至少把 formal ablation 的训练预算解释清楚。

5. **FDR**
   - 已经有 spatial permutation、receiver permutation、LR identity permutation；
   - 当前最稳说法是 group-level FDR calibrated discovery。

6. **Phase 6 SpatialSelf-specific case**
   - `CD48-CD244` 在 expanded lymph-node benchmark 中 SpatialSelf rank 1；
   - COMMOT/CellChat/SpaTalk/CellNEST 都排在 10 名以后；
   - 是 weak/context-specific；
   - 有 T cell zone enrichment、T->T context enrichment、receiver CD244 proxy、LR identity FDR 和 receiver cell-type-pair FDR；
   - 但这仍是 benchmark/real-background case，不是 native real tissue novel mechanism。

## 5. SpatialSelf 建议的 Results 叙事主线

建议不要按“我们做了很多实验”来写，而按审稿人的疑问来写：

### Result 1. SpatialSelf models LR-specific spatial communication beyond expression-distance scoring

目的：

先把方法设计讲清楚：空间图、graph wavelet、多尺度、LR identity、自监督、FDR。

必须强调：

- 不同通信机制需要不同空间范围；
- KNN 只是 candidate graph / local support，不等于 biological range；
- graph wavelet 用于多尺度空间传播；
- LR identity 是为了让不同 LR 有不同 activity pattern。

### Result 2. Mechanism-aware and real-background simulations provide realistic CCC benchmarks

目的：

回应审稿人说 simulation 太假。

图里要放：

- contact / paracrine / chemokine-like 机制；
- weak signal；
- high-expression false positive；
- real-background coordinates/expression；
- benchmark realism statistics。

### Result 3. SpatialSelf improves the weak-signal / false-positive tradeoff

目的：

核心性能 claim。

不要只放 AUROC/AUPR。要把：

- weak-signal recall；
- empirical FDR；
- top-k precision；
- high-expression FP rate；
- calibration error；

放在一起讲。

推荐语言：

> SpatialSelf improves the weak-signal/false-positive tradeoff rather than uniformly dominating every baseline on every metric.

### Result 4. Deep spatial CCC baselines and ablations separate model design from capacity

目的：

回应“没比 CellNEST/DeepTalk”和“是不是普通 GNN”。

主文至少放：

- COMMOT、CellChat、SpaTalk、CellNEST；
- DeepTalk 如果能跑，放 formal；如果不能跑，放 status table；
- no LR identity；
- no graph wavelet；
- GAT/MLP alternative；
- pseudo-label strategy。

### Result 5. Permutation testing provides FDR-calibrated CCC discoveries

目的：

这是我们和很多 deep CCC 工具差异化的地方。

图里要明确：

- 三个 null；
- p-value distribution；
- group-level discoveries；
- positive/negative controls；
- edge-level limitation。

### Result 6. SpatialSelf recovers global and region-specific communication in real-background and real samples

目的：

不要只展示漂亮空间图。

先全局：

- positive vs negative control rank；
- downstream target/proxy correlation；
- region enrichment；
- FDR；
- method comparison。

再案例：

- lymph node CCR7/CCL19/CCL21；
- GC CXCL13/CXCR5；
- Phase 6 CD48-CD244 SpatialSelf-specific weak candidate；
- tonsil/melanoma 作为扩展。

### Result 7. SpatialSelf identifies communication modules and interpretable LR structure

目的：

把文章从“LR ranking 工具”提升到“communication program 方法”。

这部分目前最好作为后续补强：

- LR embedding UMAP；
- pathway category enrichment；
- LR activity module；
- T-zone chemokine module；
- GC module；
- melanoma immune-suppression module。

## 6. 还需要补什么实验

按优先级分三层。

### 必须补强：决定文章能否站住

1. **DeepTalk formal baseline 或失败原因表**
   - Reviewer 2 明确点名 deep learning CCC；
   - CellNEST 已经有，但 DeepTalk 仍需要尽量跑；
   - 如果跑不了，要写清楚：代码、依赖、输入格式、输出层级、运行资源。

2. **真实 downstream target validation**
   - Phase 6 现在是 receptor-expression proxy；
   - 需要至少对 key LR 做 NicheNet / OmniPath / Reactome / KEGG target signature；
   - 指标：communication score vs receiver target signature Spearman + permutation FDR。

3. **benchmark realism figure**
   - library size；
   - sparsity；
   - mean-variance；
   - nearest-neighbor distance；
   - Moran's I；
   - LR expression distribution；
   - real vs simulation PCA/UMAP。

4. **formal baseline-aligned source table**
   - 所有方法同一 LR universe；
   - 同一数据；
   - 明确 missing/failed/status；
   - 不能只展示成功跑的方法。

5. **Phase 6 扩展到至少另一个真实背景**
   - 现在 CD48-CD244 是 lymph-node expanded benchmark；
   - 最好在 tonsil 或 melanoma 再找一个 SpatialSelf-specific weak/context LR；
   - 否则会被说是单个案例。

### 强烈建议补：决定文章层次

6. **CCC module analysis**
   - LR-LR activity profile clustering；
   - module spatial enrichment；
   - pathway enrichment；
   - downstream target enrichment；
   - 这能让 SpatialSelf 从 top-pair 工具变成 communication program 工具。

7. **LR embedding biological interpretation**
   - UMAP/t-SNE；
   - chemokine/cytokine/contact/ECM/checkpoint enrichment；
   - embedding distance vs predicted range；
   - contact LR 是否更短距离，secreted LR 是否更长距离。

8. **hyperparameter sensitivity**
   - KNN；
   - graph wavelet scale；
   - LR embedding dimension；
   - pseudo-label ratio；
   - distance bandwidth；
   - learning rate/dropout。

9. **runtime/scalability**
   - 1k、5k、10k、50k；
   - 如果能做 Xenium/Visium HD subset 更好；
   - 目前不要主打 100k scalability，除非真跑完。

### 可作为补充：增强完整度

10. **negative controls 更系统**
    - shuffled LR database；
    - non-permissive cell-type pairs；
    - spatially shuffled coordinates；
    - housekeeping high-expression pairs；
    - wrong-domain high-expression pairs。

11. **cross-platform real validation**
    - lymph node；
    - tonsil；
    - melanoma；
    - Xenium / Visium HD subset；
    - 无 ground truth 时用 spatial concentration、target response、known pathway enrichment。

12. **uncertainty / calibration**
    - reliability diagram；
    - calibration error；
    - null p-value uniformity；
    - discovery count under null。

## 7. 建议的最终文章标题方向

可以考虑这类标题：

1. **SpatialSelf learns LR-specific multiscale spatial communication with FDR-calibrated weak-signal discovery**
2. **Self-supervised spatial modeling of weak and context-specific cell-cell communication**
3. **LR-identity-aware graph wavelets improve weak-signal communication inference in spatial transcriptomics**

最稳的是第 1 个，因为它把三个卖点都放进去了：

- LR-specific；
- multiscale spatial；
- FDR-calibrated weak-signal discovery。

## 8. 下一阶段执行路线图

按“最能改变文章档次”的顺序，建议下一阶段不要再泛泛加实验，而是围绕四条证据链补齐。

### 8.1 第一优先级：functional validation chain

目标：

把现在的 receiver-expression proxy 升级为真正的 downstream target / pathway response。

最小可行实验：

- 选 lymph node 的 `CCL19/CCL21-CCR7`、`CXCL13-CXCR5` 和 Phase 6 的 `CD48-CD244`；
- 用 NicheNet / OmniPath / Reactome / GO / CellChatDB 构建 receiver target signature；
- 对每个 receiver cell/spot 计算 target activity；
- 计算 incoming SpatialSelf score 与 target activity 的 Spearman；
- 做 spatial / receiver / LR identity permutation FDR；
- 和 COMMOT、CellChat、SpaTalk、CellNEST 的同 LR score 做同样相关。

这条链解决的问题：

- “你怎么知道这像真的通信？”
- “自监督是不是只学到了自己的 score？”
- “真实样本没有 ground truth 怎么验证？”

主文价值：

很高。建议放主文 Fig. 7 或 Fig. 8。

### 8.2 第二优先级：DeepTalk / HoloNet / stMLnet baseline status

目标：

把审稿人最容易攻击的 baseline 缺口补上。

最小可行实验：

- DeepTalk 尽量 formal run；
- HoloNet、stMLnet、CytoSignal 如果能跑就跑；
- 如果跑不了，形成 structured status table：
  - code availability；
  - dependency reproducibility；
  - required input；
  - output level；
  - whether supports LR-pair ranking；
  - runtime/memory failure；
  - exact failure log。

这条链解决的问题：

- “为什么没有比较 deep learning spatial CCC baselines？”

主文价值：

DeepTalk 如果能跑，放主文；跑不了，主文一句说明，补充表详细列。

### 8.3 第三优先级：communication module / LR embedding interpretation

目标：

让 SpatialSelf 不只是 top LR 排名工具，而是能发现 communication programs。

最小可行实验：

- 每个 LR pair 形成 spatial activity vector；
- 计算 LR-LR similarity；
- 聚类得到 modules；
- 对 module 做：
  - spatial enrichment；
  - cell-type-pair specificity；
  - pathway category enrichment；
  - downstream target enrichment。

可展示模块：

- lymph node T-zone chemokine module；
- GC B-cell/FDC-like module；
- melanoma immune-suppression / macrophage-T-cell module；
- contact-dependent T-cell module。

这条链解决的问题：

- “LR embedding 有没有生物学意义？”
- “结果是不是一堆孤立 pair？”
- “能不能解释组织结构？”

主文价值：

如果结果清楚，适合主文后半；否则放补充。

### 8.4 第四优先级：benchmark realism and robustness polish

目标：

把 simulation 被审稿人攻击的风险降到最低。

必须补的图：

- real vs semi-synthetic library size；
- sparsity；
- mean-variance；
- nearest-neighbor distance；
- spatial density；
- Moran's I；
- LR expression distribution；
- PCA/UMAP overlap。

必须补的扰动：

- dropout；
- coordinate noise；
- ligand/receptor dropout；
- spot aggregation；
- cell-type mixing；
- irregular tissue boundary。

这条链解决的问题：

- “simulation too artificial”
- “method overfits its own assumptions”

主文价值：

benchmark realism 放主文；extended robustness 放补充。

## 9. 主文叙事建议

最终 Results 可以压成 7 个部分：

1. SpatialSelf models LR-specific multiscale spatial communication.
2. Mechanism-aware and real-background benchmarks test weak and context-specific CCC.
3. SpatialSelf improves the weak-signal/false-positive tradeoff against retained baselines.
4. Ablations identify LR identity and graph-wavelet modeling as key contributors.
5. Permutation testing provides FDR-calibrated group-level communication discoveries.
6. Downstream target validation supports biologically plausible communication in real-background and real ST panels.
7. Communication modules and LR embeddings reveal interpretable spatial communication programs.

这 7 个 Results 对应的审稿人防线是：

| Results | 防守的审稿问题 |
|---|---|
| 1 | 不是普通 expression-distance / GNN 打分器 |
| 2 | simulation 不够真实 |
| 3 | 没有证明 weak signal 和假阳性优势 |
| 4 | 模型容量 / pseudo-label bias |
| 5 | 没有 p-value / FDR |
| 6 | 真实数据只挑漂亮案例 |
| 7 | LR embedding / communication program 缺生物解释 |

## 10. 一句话总策略

SpatialSelf 不要讲成“比所有方法都强的新 GNN”。更稳、更高级的讲法是：

> 现有 CCC 方法已经能做 LR database scoring、spatial OT、LRT knowledge graph 和 GNN neighborhood modeling，但弱通信、LR-specific spatial range、高表达假阳性和统计 FDR 仍然没有被一个统一框架同时解决。SpatialSelf 的贡献是把 LR identity-aware multiscale spatial representation 和 self-supervised scoring 结合起来，并通过 realistic benchmark、deep baseline comparison、ablation、FDR 和 downstream validation 证明它能更稳地发现 weak/context-specific CCC。

这条叙事能直接回应审稿人的核心攻击：

- simulation 不真实 -> real-background + realism statistics；
- baseline 不够 -> CellNEST/DeepTalk/COMMOT/CellChat/SpaTalk；
- deep model 黑箱 -> ablation + LR embedding interpretation；
- top-K pseudo-label 偏强信号 -> weak-signal benchmark + pseudo-label ablation；
- 假阳性 -> high-expression negative controls + FDR；
- 真实数据只挑案例 -> global control ranking + downstream validation + modules。

## 参考论文入口

- CellChat, Nature Communications 2021: https://www.nature.com/articles/s41467-021-21246-9
- ICELLNET, Nature Communications 2021: https://www.nature.com/articles/s41467-021-21244-x
- NicheNet, Nature Methods 2020: https://www.nature.com/articles/s41592-019-0667-5
- SpaOTsc, Nature Communications 2020: https://www.nature.com/articles/s41467-020-15968-5
- COMMOT, Nature Methods 2023: https://www.nature.com/articles/s41592-022-01728-4
- SpaTalk, Nature Communications 2022: https://www.nature.com/articles/s41467-022-32111-8
- Tensor-cell2cell, Nature Communications 2022: https://www.nature.com/articles/s41467-022-31369-2
- DeepLinc, Genome Biology 2022: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02692-0
- MISTy, Genome Biology 2022: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02663-5
- DeepTalk, Nature Communications 2024: https://www.nature.com/articles/s41467-024-51329-2
- CrossChat, Nature Communications 2024: https://www.nature.com/articles/s41467-024-54821-x
- CellNEST, Nature Methods 2025: https://www.nature.com/articles/s41592-025-02721-3
- NicheCompass, Nature Genetics 2025: https://www.nature.com/articles/s41588-025-02120-6
- Niche-DE, Genome Biology 2024: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-023-03159-6
- Renoir, Nature Communications 2026: https://www.nature.com/articles/s41467-026-72388-7
- FineST, Nature Communications 2026: https://www.nature.com/articles/s41467-026-70528-7
- CCC benchmark, Genome Biology 2022: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02783-y
