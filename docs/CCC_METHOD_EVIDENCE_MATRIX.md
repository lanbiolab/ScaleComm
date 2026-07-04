# CCC 高水平方法论文证据矩阵

用途：把 NC / GB / Nature Methods 及以上层级的 CCC 方法论文拆成可复用的叙事证据。这里的“以上”按影响力近似包括 Nature Communications、Genome Biology、Nature Methods、Nature Genetics、Nature Cell Biology 以及同级高水平期刊；部分常被审稿人点名但不属于该层级的工具单独标为 `status/reference`。

## 1. 方法论文矩阵

| 方法 | 期刊/年份 | 方法类型 | 文章主叙事 | 关键实验证据链 | 对 SpatialSelf 的启发 |
|---|---:|---|---|---|---|
| CellChat | Nature Communications 2021; Nature Protocols 2025 | LR database + network analysis | scRNA-seq 可以系统推断 intercellular communication networks，并做 pathway / cell-group / network pattern 解释。 | curated LR database、communication probability、pathway-level network、真实数据案例、可视化生态。 | 不要和 CellChat 拼 database；要说我们解决 spatial weak signal 和 high-expression false positives。 |
| ICELLNET | Nature Communications 2021 | curated LR database + communication score | 用 expert-curated LR database 和多亚基表达规则 dissect intercellular communication。 | database、multi-subunit handling、score、reference cell-type mapping、可解释可视化。 | Methods 必须写清 LR database 和 complex handling，否则会被认为不严谨。 |
| CellPhoneDB | Nature Protocols 2020; Nature Methods database lineage | multi-subunit LR database + permutation | 可靠 CCC 推断首先需要 manually curated ligand/receptor complexes 和 permutation-based specificity。 | multi-subunit complexes、cell-type specificity、permutation test、protocol/tool ecosystem。 | 我们 Methods 必须交代 complex/multimer handling；FDR/permutation 叙事可借鉴，但要强调空间和 weak signal。 |
| NicheNet | Nature Methods 2020 | ligand-target / receiver response | 通信不能只看 LR 共表达，要看 ligand 是否能解释 receiver target genes。 | ligand-target prior、receiver target prediction、真实 immune/tumor case。 | 我们必须补 downstream target validation；Phase 6 receptor proxy 还不够。 |
| SoptSC | Nature Communications 2019 | optimal transport + stochastic process | 从 scRNA-seq 推断 lineage 和 cell-cell communication，强调 optimal transport / signaling probability。 | OT、cell fate / lineage、cell-cell communication network、developmental examples。 | 早期 OT/communication 思路；不作为主要 spatial ST baseline，但可用于引言背景。 |
| SpaOTsc | Nature Communications 2020 | spatial OT from scRNA-seq + spatial marker | scRNA-seq 丢失空间，structured optimal transport 可恢复 spatial signaling relationship。 | OT mapping、spatial signaling relationship、真实 spatial reference data。 | SpatialSelf 要强调真实空间 ST 中 LR-specific spatial activity，而不是泛泛“用了空间”。 |
| COMMOT | Nature Methods 2023 | collective optimal transport | CCC 受空间距离和 ligand/receptor competition 影响，OT 可推断 spatial signaling direction 和 downstream targets。 | synthetic benchmark、spatial map、directionality、downstream analysis、多平台真实数据。 | COMMOT 是必须认真对比的强 baseline；我们的差异是 weak/context-specific + LR identity + FDR。 |
| SpaTalk | Nature Communications 2022 | spatial LR + LRT knowledge graph | 空间 CCC 需要 ligand-receptor-target 链，不只是 LR coexpression。 | knowledge graph、LRT scoring、spatial niche/case validation、真实数据案例。 | 我们后续要把 target response 接上，否则 biological validation 会弱。 |
| Tensor-cell2cell | Nature Communications 2022 | tensor decomposition / multi-context | 单样本 pairwise 比较不够，多 context 下 CCC program 需要 tensor factorization。 | multi-context tensor、latent communication patterns、disease/stage/location context。 | SpatialSelf 后期可做 cross-tissue / module，而不是只做 single LR case。 |
| NeuronChat | Nature Communications 2023 | neural-specific CCC database | 神经系统通信有专门配体/受体和神经递质知识，通用 LR database 不够。 | neural signaling database、brain datasets、network analysis、region/cell-type communication。 | 如果做脑数据或神经系统，需要 tissue-specific LR prior；当前可作为 LR identity/category 重要性的旁证。 |
| DeepLinc | Genome Biology 2022 | deep generative graph model | 从 single-cell spatial transcriptomics de novo 重构 cell interaction landscape。 | VGAE latent interaction network、spatial domain、case validation。 | 深度模型文章会被要求解释 latent representation；我们需要 ablation 和可解释性。 |
| MISTy | Genome Biology 2022 | explainable multiview spatial ML | spatial omics 关系来自不同空间视野，需要 explainable multiview framework。 | intraview/paraview/juxtaview、spatial context、feature importance、clinical relation。 | graph wavelet 的多尺度叙事可以借鉴：不是单一 KNN，而是多尺度空间贡献。 |
| CCC benchmark | Genome Biology 2022 | benchmark / evaluation | CCC inference 方法需要用空间信息和统一框架评估，不能只靠案例。 | multiple methods、spatially-informed evaluation、benchmark metrics。 | 我们要有 benchmark realism、统一 LR universe、baseline status/failure reasons。 |
| DeepTalk | Nature Communications 2024 | subgraph graph attention | single-cell resolution ST 需要 subgraph-based GAT 捕捉局部结构与 CCC。 | deep spatial model、benchmark/real cases、cell-cell interaction visualization。 | Reviewer 点名 deep baseline；能跑就 formal compare，跑不了要 status table。 |
| CrossChat | Nature Communications 2024 | hierarchical global/local CCC | CCC 有 global 和 local hierarchical structure，不能只看 pairwise top LR。 | hierarchy、local/global communication patterns、multi-dataset examples。 | 我们要补 communication module / program analysis，避免只是 top-pair ranking。 |
| LIANA+ | Nature Cell Biology 2024 | consensus CCC + multi-view framework | CCC 推断需要整合多个工具、空间邻域、intracellular signaling 和多模态视角。 | consensus scoring、method harmonization、spatial/multi-view analysis、pathway/TF linkage。 | 可作为 consensus/meta-baseline 或讨论对象；我们要强调 FDR-calibrated weak-signal discovery 这个差异点。 |
| Niche-DE / niche-LR | Genome Biology 2024 | niche response + ligand-receptor | 先找 niche-associated genes，再用 ligand-target/LR prior 解释 cell-type-specific niche response。 | niche-DE、ligand/receptor/target integration、spatial niche validation。 | 我们后续要从 LR score 升级到 niche response / downstream response。 |
| NicheCompass | Nature Genetics 2025 | graph deep learning communication programs | graph deep learning 可学习 signaling-event encoded niche embeddings，定量刻画 niches。 | interpretable cell embeddings、communication programs、cross-sample alignment、atlas-scale cases。 | SpatialSelf 如果要高层次，必须讲 LR module / embedding / cross-sample program。 |
| CellNEST | Nature Methods 2025 | attention relay network | ST 中 CCC 不只是两点通信，还包括 attention-based relay networks。 | attention mechanism、relay network、spatial transcriptomics benchmark/cases。 | 已纳入 baseline；还要把“relay/attention”与我们的 graph wavelet 差异说清。 |
| Renoir | Nature Communications 2026 | spatial ligand-target activity | 需要在空间 niche 中刻画 ligand 对 downstream target 的影响。 | ligand-target neighborhood score、communication niches、pathway activity、Visium/Visium HD/Xenium/CosMx、实验验证案例。 | 这是我们后续最该对齐的 functional validation 方向。 |
| FineST | Nature Communications 2026 | histology + ST contrastive learning + LR | ST 稀疏和分辨率限制影响 CCC，histology foundation model 可提升 nuclei-resolved LR pattern discovery。 | histology-ST fusion、expression imputation、cell type prediction、LR pattern discovery、VisiumHD/Xenium。 | 高分辨率和 histology 融合是加分方向，但不是当前核心；可作为未来扩展。 |

## 2. 常被点名但不应夸成 NC/GB 主证据的工具

| 方法 | 状态 | 为什么仍要考虑 |
|---|---|---|
| HoloNet | Briefings in Bioinformatics 等非 NC/GB 主线 | 审稿人可能点名 spatial CCC / higher-order network；适合 status baseline 或补充讨论。 |
| stMLnet | 期刊层级和输入输出需再核查 | 可作为 spatial multi-layer network 参考；如果不能 formal run，写明原因。 |
| CytoSignal | 需要核查具体版本和输出层级 | 如果能输出 LR/cell-type score，可作为补充 baseline；否则 status table。 |
| NICHES | 常用 spatial CCC 工具，但不属于本调研主层级 | 用户/审稿人可能期待；保留 status/failure reason。 |
| MultiNicheNet | bioRxiv / package | 多样本 differential CCC 叙事很强，但不要写成 NC/GB 以上正式方法论文。 |
| CytoTalk | Science Advances 2021 | de novo signal transduction network 思路值得借鉴，但不是 NC/GB/Nature 系主线。 |

## 3. 高水平文章的共同“故事模板”

高水平 CCC 方法文一般不是这样写：

> 我们提出一个模型，AUROC 更高，所以发表。

而是这样写：

1. **定义一个旧方法解决不了的生物/统计缺口**
   - 没空间；
   - 没 downstream response；
   - 没多 context；
   - 没 high-resolution scalability；
   - 没 FDR；
   - 高表达假阳性多；
   - deep model 黑箱。

2. **每个模型组件都对应该缺口**
   - LR database -> biological prior；
   - target prior -> functional response；
   - OT -> spatial allocation；
   - GNN/attention -> local subgraph / relay；
   - tensor/multiview -> context；
   - permutation -> FDR；
   - module/embedding -> interpretability。

3. **benchmark 是独立证据，不是自己喂自己**
   - synthetic + real-background + real data；
   - realistic statistics；
   - multiple communication mechanisms；
   - weak signal；
   - false positive high-expression controls；
   - baseline status table。

4. **真实数据先全局，后案例**
   - global positive/negative controls；
   - target response correlation；
   - region enrichment；
   - cell-type specificity；
   - pathway/module enrichment；
   - selected LR map 只作为最后展示。

5. **深度方法必须证明不是黑箱**
   - no LR identity；
   - no graph；
   - no distance/expression；
   - MLP/GAT/GCN alternative；
   - pseudo-label ablation；
   - embedding/module interpretation。

## 4. SpatialSelf 的最佳定位

一句话：

> SpatialSelf learns LR-specific multiscale spatial communication activity and provides FDR-calibrated discovery of weak/context-specific CCC.

不要说：

> SpatialSelf beats all CCC tools everywhere.

应该说：

> Existing CCC methods cover LR database scoring, ligand-target response, spatial optimal transport, and graph-based neighborhood modeling, but weak/context-specific communication remains difficult because high-expression LR pairs create false positives and fixed spatial rules cannot represent LR-specific ranges. SpatialSelf addresses this gap by combining LR identity-aware graph wavelet modeling, self-supervised scoring, and permutation-based FDR calibration.

## 5. SpatialSelf 还需要补的关键实验

### 必须补

1. **Downstream target validation**
   - 用 NicheNet / OmniPath / Reactome / KEGG / CellChatDB target sets；
   - 计算 incoming score vs receiver target signature；
   - 和 COMMOT / CellChat / SpaTalk / CellNEST 对比；
   - permutation FDR。

2. **DeepTalk formal baseline 或 structured failure table**
   - DeepTalk 是 Nature Communications 2024；
   - Reviewer 很容易继续追问；
   - 跑不了也必须有可审计原因。

3. **Benchmark realism figure**
   - library size；
   - sparsity；
   - mean-variance；
   - nearest-neighbor distance；
   - Moran's I；
   - LR expression distribution；
   - real vs semi-synthetic PCA/UMAP。

4. **Phase 6 SpatialSelf-specific case 扩展**
   - 当前只有 lymph-node expanded benchmark 的 `CD48-CD244`；
   - 最好再在 tonsil 或 melanoma 找一个 SpatialSelf rank <= 10、所有 baseline > 10 的 weak/context LR；
   - 避免被说成单个幸运案例。

### 强烈建议补

5. **CCC module analysis**
   - LR activity profile clustering；
   - module region enrichment；
   - module pathway enrichment；
   - receiver target enrichment。

6. **LR embedding interpretation**
   - functional category enrichment；
   - embedding distance vs signaling range；
   - contact vs secreted LR separation；
   - chemokine/cytokine/checkpoint/ECM modules。

7. **runtime / scalability**
   - 1k、5k、10k、50k；
   - high-resolution subset；
   - wall-clock、CPU/GPU memory、candidate edges。

## 6. 主文推荐结果结构

1. SpatialSelf models LR-specific multiscale spatial communication.
2. Mechanism-aware and real-background benchmarks test weak/context-specific CCC.
3. SpatialSelf improves the weak-signal / high-expression false-positive tradeoff.
4. Deep baseline comparison and ablations separate model design from capacity.
5. Permutation testing provides FDR-calibrated group-level discoveries.
6. Downstream target validation supports biologically plausible communication.
7. Communication modules and LR embeddings reveal interpretable spatial programs.
