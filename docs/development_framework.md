# Transformer Disentanglement 研发框架

## 1. 总目标

识别蛋白质 Transformer pairwise statistic 中的不同来源，并建立一种可验证、可泛化的分离方法：

\[
T_\theta(X)
= B_{\mathrm{position}}
+ B_{\mathrm{one-body}}
+ B_{\mathrm{motif}}
+ B_{\mathrm{phylogeny}}
+ I_{\mathrm{interaction}}
+ \epsilon.
\]

其中最终希望保留的是能够预测结构接触、突变 epistasis 和 fitness 的交互项
`I_interaction`，而不是简单删除低秩或最大本征模态。

## 2. 核心科学问题

1. Transformer 中是否存在可重复识别的 nuisance component？
2. nuisance 主要来自位置、PSSM、单序列 motif、phylogeny，还是 softmax normalization？
3. 真实 interaction 是否能够通过 intervention/null subtraction 提取？
4. 提取出的 interaction 是否比 attention、APC 或 contact head 更能泛化？
5. 能否把事后分离改成训练期约束，并改善模型表示或生成能力？

## 3. 研究原则

- 在模型函数层面定义 interaction，不把某个参数矩阵天然解释为 coevolution。
- 使用 matched null 和 intervention 建立因果含义，不只看相关性或低秩性。
- 先验证现象，再设计正则项。
- 主要结论必须跨 protein family、模型规模和随机种子成立。
- attention head 选择必须在训练家族完成，在独立家族评估。
- supervised contact head 只作为工程参照，核心结论应由无监督 statistic 支持。

## 4. 六层研发结构

### Layer A：数据与 null generator

输入：真实 MSA、query sequence、结构、DMS/fitness、可选 phylogenetic tree。

需要实现：

- `real`：真实 homolog ensemble；
- `pssm_null`：按每列边际独立采样；
- `column_shuffle`：逐列置换，严格保持有限样本 PSSM；
- `global_composition`：只保持全局氨基酸组成；
- `query_repeat`：重复单条 query；
- `position_permutation`：统一打乱列并记录逆映射；
- `phylogeny_null`：沿树模拟独立位点演化；
- `potts_positive_control`：带已知 pairwise coupling 的模拟数据。

输出：统一格式的数据包、seed、depth、PSSM、gap rate、tree 和 ground truth。

### Layer B：模型 statistic extractor

所有 extractor 使用同一接口，输出位置级矩阵或 `L x A x L x A` tensor：

- post-softmax attention；
- pre-softmax `QK^T / sqrt(d)`；
- supervised contact head；
- categorical Jacobian；
- double-mutant finite difference；
- pseudo-log-likelihood epistasis；
- hidden-state covariance 和 cross-covariance。

优先级：categorical Jacobian > double-mutant score > pre-softmax logits > attention。

### Layer C：现象诊断

每种 statistic 都计算：

- spectrum、singular values 和 effective rank；
- leading vector 与 entropy/conservation/gap/surprisal 的相关性；
- real-null map correlation；
- APC 前后变化；
- layer/head 稳定性；
- MSA depth 和模型规模依赖；
- contact、epistasis 和 fitness precision。

这一层只回答“有什么”，不进行训练期修改。

### Layer D：disentanglement 方法

按可解释性从高到低依次研究：

1. Null subtraction

   \[
   I(X)=T(X)-\mathbb{E}_{X'\sim Q_{null}}T(X').
   \]

2. Amino-acid zero-sum gauge

   \[
   J^{int}_{ij}=C_AJ_{ij}C_A,
   \quad C_A=I-\frac{1}{A}11^T.
   \]

3. Known-nuisance projection

   使用 entropy、conservation、gap rate、surprisal、position basis 构造 `Z`，将 statistic
   投影到 `span(Z)` 的正交补。

4. Functional ANOVA / Hoeffding decomposition

   相对于指定 reference distribution 分离 one-body、pairwise 和 higher-order effect。

5. Robust low-rank + structured residual

   只作为比较方法；低秩不能自动解释为 nuisance。

6. Learned nuisance subspace

   使用跨家族训练的 encoder/adversary 学习 nuisance，但必须通过 null intervention 约束语义。

### Layer E：训练期干预

只有 Layer D 在 held-out family 成功后才进入本层。

候选方案：

- nuisance-subspace penalty；
- Jacobian interaction regularization；
- null consistency loss；
- real-null contrastive objective；
- one-body branch 与 interaction branch 的显式双分支结构；
- LoRA/adapter 微调，避免直接破坏基础模型。

不优先：直接惩罚 attention 最大本征值，因为 softmax 已经固定了平凡 Perron 模态。

### Layer F：生物学验证

验证任务按强度排列：

1. 长程结构接触；
2. experimental double-mutant epistasis；
3. single-mutant effect；
4. protein stability/fitness；
5. sequence generation 的 diversity、foldability 和 novelty；
6. 必要时开展小规模实验验证。

## 5. 数据集分层

### Tier 0：开发集

- 3CNBA；
- 用途：代码、null、可视化和 metric 校验；
- 禁止用于最终选择方法或 head。

### Tier 1：校准集

- 从 383-family benchmark 中选 20-30 个家族；
- 按长度、MSA depth、contact density、fold class 分层；
- 用途：选择 statistic、null 和超参数。

### Tier 2：主 benchmark

- 383-family benchmark；
- 固定 train/validation/test family split；
- 所有模型选择在 test 之前冻结。

### Tier 3：外部生物学数据

- double-mutant/DMS 数据；
- GA/GB、chorismate mutase、adenylate kinase；
- 不同数据库和不同物种来源的外部分布测试。

## 6. 模型矩阵

第一阶段：

- ESM-2 8M：快速开发；
- ESM-2 35M：规模复现；
- ESM-2 150M：主模型候选。

第二阶段：

- MSA Transformer；
- ESM-2 更大版本；
- 至少一个不同训练体系的蛋白语言模型。

模型比较必须区分：

- 单序列模型参数中存储的 evolutionary statistics；
- 当前输入 MSA 中即时提取的 coevolution；
- supervised structure/contact head 注入的结构先验。

## 7. 关键判定门槛

### Gate A：现象存在

- real-null residual 在 held-out families 上显著高于 contact prevalence；
- 至少 70% 测试家族方向一致；
- bootstrap confidence interval 不跨越 null baseline。

### Gate B：语义成立

- PSSM-null、column-null、global-null 和 phylogeny-null 给出可解释的层级变化；
- interaction residual 在破坏 pairwise dependency 后消失；
- 保留 pairwise coupling 的 positive control 能被恢复。

### Gate C：方法有用

- held-out long-range contact precision 提升；
- epistasis/fitness 预测提升；
- 不依赖 test-family head selection；
- 不显著损害 MLM likelihood 和 marginal calibration。

### Gate D：训练期方法值得继续

- 至少两个模型规模和两个外部任务获益；
- 改善不能只来自 supervised contact labels；
- 计算成本和显存开销可接受。

## 8. 工作包

### WP1：Benchmark 基础设施

- 安全转换 383-family pickle；
- 固定 family split；
- 建立数据 manifest、hash 和缓存；
- 输出统一的 contact/DMS evaluator。

### WP2：Null library

- 完成六类 null；
- 验证边际、pairwise correlation 和 phylogeny 属性；
- 建立 null 单元测试。

### WP3：Statistic atlas

- 对 attention、contact head、logits、Jacobian、double-mutant score 建图谱；
- 输出 layer/head/model/family 维度的数据表。

### WP4：Post-hoc disentanglement

- 比较 null subtraction、gauge、projection、ANOVA 和低秩方法；
- 在 validation families 选择方案；
- 在 test families 冻结评估。

### WP5：训练期方法

- 从 adapter/LoRA 开始；
- 实现 null consistency 和 nuisance penalty；
- 测试是否提高结构、epistasis 和生成质量。

### WP6：机制解释与论文

- 形成从经典 APC 到 Transformer residual 的统一叙事；
- 明确哪些结论是统计分离，哪些具有生物物理含义；
- 整理主要图、消融实验和负结果。

## 9. 近期执行顺序

### Sprint 1：从单家族走向多家族

1. 安全转换 383-family 数据。
2. 选择 20 个分层家族。
3. 复现 ESM-2 8M/35M 的 real-column-null residual。
4. 冻结一个跨家族 attention-head selection 方案。

### Sprint 2：无监督 interaction

1. 实现 categorical Jacobian。
2. 实现 double-mutant finite difference。
3. 比较其 null residual 与 supervised contact-head residual。
4. 确定主 statistic。

### Sprint 3：null 语义与 phylogeny

1. 实现 phylogeny-null。
2. 区分 entropy、phylogeny 和 motif background。
3. 完成 positive-control simulation。

### Sprint 4：方法与训练

1. 比较 null subtraction、projection 和 ANOVA。
2. 在 held-out families 完成 Gate C。
3. 再决定是否开发训练期 regularizer。

## 10. 当前最重要的下一步

不是继续优化 3CNBA，也不是直接训练新 Transformer。

当前最高优先级是把 `real - matched null` 的接触富集结果扩展到独立 protein families，
并用 categorical Jacobian 证明该现象不依赖 supervised contact head。

