# Day 2：深入炼丹炉——大模型训练核心流程与源码全解析 (Pretrain & SFT)

**学习目标**：面向志在通透大模型底层训练机制的极客，今天我们将顺着数据流入大模型血管的路径（Data Pipeline）顺流而下！了解人类的杂乱文本如何被“粉碎、清洗、打包”送入模型（**Tokenizer 与 Dataset**），并亲手解剖大模型成长的两个最核心、耗时最长的脚本（**预训练 Pretrain 与 有监督微调 SFT**）。

学完今天所有超硬核代码块的内容，你将掌握为什么大模型在训练时对系统设定的废话“视而不见”（`-100`的终极掩码奥义），以及它是如何一步步用数学公式跨越人工智障，走向听话的智能的。

---

## 第一部分：第一道大门 —— Tokenizer（分词器）的诞生

大语言模型（LLM）再神通广大，它的本质也是一个只会处理矩阵数字的瞎子。计算机是无法理解人类诸如“我爱编程”这种高维抽象符号的。
在把文本喂给昨天我们手捏出来的那个巨大的 Transformer 之前，必须有一台**“高效粉碎机”**，把长句子切分为一个个能在字典本里找到固定编号的小碎块——这叫 **Token**。

### 1.1 为什么一定要切碎成 Token？
大家初学时通常会有很大疑惑：
- **方案A（粉碎成最基础字母/单字）**：把英语全按 a 到 z 的 26 个字母切，中文按单字切。
  - *致命缺陷*：如果按字母切，"hello" 算作 5 个独立的字。那么一篇文章的“序列长度（Seq_len）”会变得无穷无尽！我们昨天讲过注意力层可是要算序列的 $O(L^2)$ 乘法的！这么搞显存直接炸穿。且字母缺乏组合起来浑然天成的局部语感。
- **方案B（按完整词库切）**：把 "notebook", "apple" 甚至 "ChatGPT" 全按单独一个完整词存进词典。
  - *致命缺陷*：人类每天都在发明新词（例如“绝绝子”、“VibeCoder”）。世界上光英语单词和衍生词就有几百万个。如果在最后输出预测那一层搞一两百万维度的全连接大盘子打分，模型参数不仅爆炸，且绝大部分罕见词十年也猜不中一次。

**终极杀招：BPE（Byte Pair Encoding，字节对编码）算法**
现今乃至 OpenAI 全家桶都在用的绝对主流方案！
原理就是“**最爱抱团的凑一对**”。它在预训练之前，先扫描百万级正常人类在讲的语料：
1. 发现 `e` 和 `r` 总是一起出现，它就合并成 `er` 收进词表；
2. 发现 `h`, `e`, `l`, `l`, `o` 总是抱团，干脆把 `hello` 焊死铸造成1个独立的 ID 放进表里；
3. 对于你临时现造的极其罕见的“qwerta”，词表不认识，它就无情地原路打散退化成 `q`, `w`, `er`, `t`, `a` 几个子零件的 ID，保证**永远不会出现生僻字（Unkown）死机无法编码现象！**。因此 BPE 被称作“智能原子粒度切分法”。

### 1.2 MiniMind2 是如何从零训练自己的分词器的？
打开源码中的 `trainer/train_tokenizer.py`，作者利用 HuggingFace 底层极速工业级工具 `tokenizers` 包：

```python
def train_tokenizer(data_path, tokenizer_dir, vocab_size):
    # 【核心构建】实例化一个基底为纯正的 BPE 模型的分词器大盘框架！
    tokenizer = Tokenizer(models.BPE())
    
    # 将输入的生生人类字符串，最先过一遍 ByteLevel，强制退化为最底层的 UTF-8 字节十六进制元认知。
    # 这样连外星文或者 emoji 表情都能被拆分成机器能算的字节码，彻底断绝无法编码的乱码 OOV(Out of vocab) 噩梦。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, # 极其轻薄！MiniMind定下了全词典仅仅 6400 的袖珍参数池。
        # ⬇️【非常关键的特权天龙人 Token 钦定】⬇️
        special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 吞吐数以万计的数据，开始疯狂暴力搜刮频率最高的那些文本碎片
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    # 训练后，通过特权将其死死绑定在最前面的几个黄金坑位，以保证它们雷打不动。
    assert tokenizer.token_to_id("<|endoftext|>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2
```

**【面试必考】这三个神圣不可侵犯的特权标记（Special Tokens）是做什么用的？**
- **0 号 `"<|endoftext|>"`**：万物归寂符/空白填坑符。它有两个绝对使命：一是如果回答完了，大模型会在尾部生成这个字，外部代码见之立即掐断循环（结束标志 EOS）；二是在训练中如果规定要用长度为512长箱子装题目，你的题目只有10个字，剩下的空隙就是拿这个 0 号数字去无情填充满做滥竽充数用的（填充标志 PAD）。
- **1 号 `"<|im_start|>"`** 与 **2 号 `"<|im_end|>"`**：这俩是只有进入微调时期，为了教会大模型**什么叫“对答如流”**而专门定制的一个“信封包裹封皮”。（也就是常说的 ChatML 格式的始与终界碑，用它们来圈住每一段人类的提问或者是 AI 的回答）。

---

## 第二部分：流水线集装箱封装厂 —— Dataset 魔法构造

我们手里拿着数字，现在我们要进入 `dataset/lm_dataset.py`，看看咱们炼丹的引火柴是怎么经过打捆包装劈砍，最后符合训练要求的！
这里有预训练和 SFT 两个集装箱。

### 2.1 史前大预训练的数据打包 —— `PretrainDataset`
预训练阶段模型是个只会“阿巴阿巴”的巨婴，只要是有字的百科全书，只要能预测顺延说下一个字它就算是进步，完全不在乎是否符合“人类对话的礼仪”。
这里处理的通常是几百 G 的爬虫抓取文本大混战。

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        # ... 初始化等准备配置过程
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        
        # 1. 切丝：拿出一长篇维基百科古文，动用上面练好的 tokenizer 切成一大串的数字。截断到最大限制以内保证塞得进咱们规定的 512 的集装箱。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        
        # 2. 戴帽穿鞋：为了明确大文段的界限，用特殊的启停符给开头结尾钉上装订线
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # 3. 填空：如果截断之后发现还不够 512 的箱子宽，那就拿出万能的 pad_token_id（就是0）从后往前一路猛填（Padding）直到刚好凑够 512 的铁板一块！
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 4. 🔥【划重点！造考卷：填上神灯级魔法无视数 `-100`】
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        # 考卷题目（输入内容）和 标准改卷答案（对应的Label）就这么配对了！
        return input_ids, labels
```

**【神级面试护城河：为何要填 `-100`？】**
> 我们要求系统大张旗鼓去预测，如果一句话只有十个字，后面有 502 个 `0`（补白废位 Token）。在这 502 个 `0` 后面瞎预测下一个数字也是 `0` 的话，大模型实际上是在为了这部分“没有丝毫人类知识密度的破烂边角料区”大量空耗算力，而且它如果一不小心算成了别的字，大模型内部的计分员（`Loss`）就会把总分往狠里扣！大模型会陷入为了填空白而在学习的自我内耗，丧失了去钻研那 10 个蕴含真理文字的热情。
> 
> **解药**：PyTorch 最强大的扣分黑盒 `CrossEntropyLoss()` 天生留有一道“赦免暗门”：它内部固定参数叫 `ignore_index=-100`。
> 上面代码干的就是一件事：在这个标签答案（Label）里，凡是那些咱们用来硬凑进去填空的垃圾尾巴区域，统统把它画上红圈涂改液涂抹成 `-100`。
> 分数判官（Loss 函数）走到 `-100` 立刻两眼一翻选择失明，这部分不奖不罚直接溜过！这就极大保全了咱们算力的纯粹与准确！

### 2.2 SFT 教义微调的数据打包 —— `SFTDataset`
经过前面几十天的预训练炼丹之后，模型已经成了个“语言流浪接龙学究大神”，但你问它“地球有多大”，它可能会发疯似的回答“地球有多大？太阳有多广？”，因为它只想进行续写本能。
为了让它老实成为**一问一答的恭维机器人**，我们就要拿出包含数百万套 `[你是一个助手]->[用户发问]->[回答标准答案]` 的 QA 文件进行**有监督微调（SFT）指令对齐特训！**。

下面这就是当今 AI 界绝活中的绝活：**滑动窗口掩码免死金牌！**

```python
class SFTDataset(Dataset):
    # 初始化过程准备这那省略... 咱们重点关注考场改卷答案生成！
    
    def generate_labels(self, input_ids):
        # 1. 上来直接无情屠杀：发卷子先把整张试卷包括人设、问题、甚至全部内容全涂改成 -100 不计分的失明屏蔽码！
        labels = [-100] * len(input_ids)
        i = 0
        
        # 2. 拨云见日循环往后摸排侦查：这长长的一段里究竟有没有我要找的 Assistant 小红花起跑线？
        while i < len(input_ids):
            # 看看这当前的几个字连合体是不是咱们上面讲过的信皮印记：“<|im_start|>assistant”
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                
                # 开始沿着答题区继续往后摸，一直摸到遇见收尾印记：“<|im_end|>”
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 3. 🎯【真正惊艳时刻】：把起终点里“助手亲自吐露的精华回答”抠出来！
                # 只有这部分高光答案，才配拥有真金白银的 Label 数字底子，把它们原封不动的盖回去显影！
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                
                # 收队，这一段搜寻完毕直接越过回答末尾去接着往后侦查下一趴
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        
        # 神工鬼斧！最后吐出来的标签串，将是一个除了 [真正要训练去预测的回答部分] 能带分数印记外，
        # 余下所有的诸如系统规则 "You are a helpful.." 或者 用户废话 "今天天气..." 全是免死牌 -100 的终极定向考卷！
        return labels
```

**🔥【为什么微调的这段逻辑如此重要？】**
如果 SFT 不这么费劲巴拉做这段找皮卡丘指针遮掩（Masking）的体操，而是像第一部分预训练一样全部塞进去记分。大模型就会把所有人类打下的人设背景、人类闲聊磕牙瞎打的错字，通通奉为神谕去拼死学它们的生成概率。
这么做模型会极速“被带歪崩溃”，最后完全忘却自己。通过遮蔽法，它眼里只有：**看到前面那一堆废墨水作前提铺垫的时候，我如何尽最大努力去精确复现这唯一的标准段红线答案才最完美！**

---

## 第三部分：炼火朝天的终极炉鼎 —— Pretrain & SFT 训练代码环大起底

包装都完成了。现在我们进入整个项目的最终目的深渊，点开 `trainer/train_pretrain.py`。这是大模型之所以强大的算力燃烧之地。由于 `train_full_sft.py` 跟它只有极其小的数据导入差异，代码骨架如出一辙，所以放在一起看！

### 3.1 `train_epoch` 的狂暴更新五部曲

每一轮都在将成百上千条知识推入神坛更新参量！

```python
def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    start_time = time.time()
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        
        # 将上面包好的沉重数据铁箱推进专门干猛活的 GPU 炼丹室 (cuda:0 等设备)
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        
        # 1. [学习率调度]：极其优雅的 Cosine 余弦退火学习率
        # 根据我们现在进行到几点几分了，去衰减曲线图上爬那个学习率的点。前期先热身爬坡（Warmup），后面稳稳滑落平铺，防止模型学到后期脑抽抖动忘记真理。
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2.【显卡内存救星级外挂：AMP 混合精度包裹场（AutoCast）】
        with autocast_ctx:
            # 放行数据到昨天造好的那个巨大堆叠 Transformer(模型)深渊中去进行千山万水的跋涉和多层悟道运算（前向传播 Forward）
            res = model(input_ids, labels=labels)
            
            # 它历险到底端拿出来的不仅仅有它懵懂猜测下一个字扣分记录册 (loss)，如果开启了MoE混合专家外挂，还有可能带着由于专家偏台偏爱导致的工作量惩罚罚款项 (aux_loss)！两兄弟相加才是总惩罚值！
            loss = res.loss + res.aux_loss
            
            # [🔥工程绝命保护法宝 1]：梯度穷人累加缩影 (Gradient Accumulation)
            # 因为我们后面的反向传播是每次算出一个分段梯度累加着叠在一起。为了之后等除出整体统一做最后更新时平均力度不偏颇，先提前把每一次的分数给它按将要累加的总段数平摊稀释化！
            loss = loss / args.accumulation_steps

        # 3. 释放神迹：反向传播（Backward）大追溯！
        # 让误差如倒悬长河般溯游而上，挨个揪出沿路几千万参数：刚才这一道题答这么差，你这个当神经元节点的权重该担多少责任！统统记在我将要加总求和的小本子（Grad）上准备批斗！
        # scaler放大器：为了防止那可怜巴巴浮点16的半精度在回推时数值太小变成虚无绝育，被专门设计的缩放放大器给强行托举放大不致丢了精度。
        scaler.scale(loss).backward()

        # 等到在这个平稳度日子的步子里，苟且偷生循环终于满积累了一大批次，咱们终于能凑出一次全方位宏观的高配更新批次的时候！
        if (step + 1) % args.accumulation_steps == 0:
            
            # 还原那个在求导数时为了保命强行虚高托举的显微镜（解封真实梯度参数）
            scaler.unscale_(optimizer)
            
            # 4. [🔥工程绝命防护法宝 2]：梯度粗暴一刀腰斩法 (Gradient Clipping / clip_grad_norm)
            # 大模型是个恐怖的数值滚雪球厂。如果在某个不恰当的批次里，有几个字极为离谱相差十万八千里从而引发了狂暴的巨大修改怒火，任由这些极端逆差打入参数层直接会将几周的炼丹沉淀完全炸毁归为混沌（NaN崩溃）！
            # 有了这一手绝斩阀门卡口：不管你多错多离谱产生的调整值怒火怨气有多么排天倒海，咱们的惩戒力度绝对不能超越这1.0个数值红线标准，一切强行封顶！稳如泰山防止一切数值翻车事故！
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 5. 走你！（Optimizer Step！）
            # 集结着批斗小本上的累加梯度控诉，经过统筹和阀门压制后。优化器大总管（如 AdamW 超级迭代官）正式号令分布在模型各个边角的参数：朝正确的山头挪近步履吧！
            scaler.step(optimizer)
            scaler.update()

            # 6. 大清洗除魔卫道！把这次记恨复仇的满页积累小本上的梯度全部连根焚毁冲散置空（zero_grad），满怀轻装从零开始准备进下一局！
            optimizer.zero_grad(set_to_none=True)

        # 7. 打印打表或者报告给远端监控的包工头管家控制台 (wandb)，告知我们的loss是否如期下降心安理得
        if step % args.log_interval == 0 or step == iters - 1:
            # 省略打印与保存代码...
            pass
```

### 3.2 高阶分布式训练战线（DDP）

随着规模从几千万走到几十亿参数，单卡就算累吐血跑 200 年也炼不出模型了。
在文件的最底部代码调用模块部分，你可以看到真正的神级黑卡集团大作战 —— **分布式训练 DistributedDataParallel(DDP)** 组建网阵的过程。

```python
    # 一开始分配占阵符，各个子进程卡得到通知说你今天排位在几号地
    local_rank = init_distributed_mode()
    
    # 【黑丝绒地毯级神包装】
    if dist.is_initialized():
        # 这里特别标记除了两个天选之子绝不要加入显卡分布更新战争大军中——对啦，他们就是你昨天学的 RoPE 的位置圆圈频率表！他们是死参数（Constant不用更新梯度的恒常护身符），无需去掺和运算梯度的战壕里
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 用 DDP 极其厚重伟岸的无死角保护外壳大膜把咱们原本娇小的单机版 Transformer 尸体层层裹起保护，强行转化接入全分布式并行矩阵战舰网！
        model = DistributedDataParallel(model, device_ids=[local_rank])
```

到了这里，只要利用如 `torchrun --nproc_per_node 8` 等起兵符指令发动！
在一瞬间，这块大网布就会将你的整个模型极其对称完美地给每一张卡各自复制投胎一份。接着用一个能极其均匀把几十个 G 的大文本精准无重叠平均丢给这八个人各自发包派送的数据撒种机发牌员（`DistributedSampler`），让他们同时计算属于自己的那部分！
并在反向求导的终点，靠着底层极其精密的 NVLink 和光纤通讯，硬汉般把八份梯度进行暴力平均综合，然后命令所有人同时踏出这完美而宏观的优化一脚！！

---

## 第四部分：Pretrain （野蛮生长） 对抗 SFT （规训对齐）的终极异同大赏

尽管这两位“一气化三清”共享着底层完全相同的流水线和运算机器，但从 `train_pretrain` 和 `train_full_sft` 命令行配制和参数调性的异同上，我们能直击炼制大模型的最高宏观核心心法！

| 对抗维度 | 第一炉火炼：Pretrain 混沌洗脑填骨期 | 第二炉火炼：SFT 全参数微调雕花指令期 |
|:--------|:-------------------------------------------------------|:---------------------------------------------------------|
| **所食丹药用料** | 上知天文下知三字经黄页的**混乱随机生食数据原片语段**。只要是个顺流子文字就行。 | 极具格式洁癖的，动用模板精雕细琢前后套皮标注成**聊天QA体**。 |
| **打分红线(Loss)**| 要求记住每一个字的续写律动规则（全文算入严厉计分损失）。 | 只要求把 `Assistant` 出声发言的部分算分考过就行，闲杂人等提问一律屏蔽（`-100`加身掩藏大法）。 |
| **初始大局学习率**| `5e-4` (**大开大合！**) 我原本如同一张白纸极其愚钝的人工智障弱胎，此时正是虚心吸取天地一切大框架知识灵气之际，步子迈到多快大都不过分！ | `1e-6` (**极小极慢拉微操！细致入微的神经微调**)。千万不敢让步子大扯着蛋！老子好不容易上一环花了几千块电费背会的全世界庞大知识库如果此时用大力冲刷会被统统覆盖遗忘掉！这叫【灾难性遗忘症防范大关卡】！所以微观调整只为让性格变得听话而绝不可以动了知识本身储备底骨！ |

---

## 🚀 今日终极顿悟结业：炼丹炉外的你该何去何从？

如果你读到了这里并能倒背出这些坑点为什么存在、这个小命令代码这行起到了什么护命效果，那么恭喜你！在绝大多数还在问“怎么调 Prompt 的外围调包侠新手里面”，你已经属于能直视机器显存火盆内部运行，手握造物主大脉搏的高段位初醒者了！

**✅ 请你最后一次在心中自查今日的三大护城河认知神迹是否铭刻：**
1. **为什么要费尽心机搞 BPE Tokenizer** 而不干干脆脆按单独英文字切断喂饱？
2. 在庞大的训练标签答案纸堆里，这个如魔法如禁咒般存在穿插于废话填充间的 **魔数 `-100` 究竟是为了解救什么绝境**？
3. 在深不见底的数据大轮回 `train_epoch` 脚本战车上，**梯度截断限制** 与 **穷人法宝大累积除商大法** 是为了保咱们什么命格无忧？

> 💡 **上机实战演练发令枪任务：** 现在点开你的控制台。既然我们已手无寸铁硬刚解梦了每一行来龙去脉，那就去真正的将这小东西在本地电脑轰然点燃发动起来吧！
> 在工程的目录下跑起（可以把 batch_size 调低迎战显存压力）：
> `python trainer/train_pretrain.py --epochs 1 --batch_size 2`
> 
> 去看看那个真实跳动的 `Epoch[xxx/xxx] loss: xxx ` 和 `eta_min: xxx` 在屏幕上随着时间的推进缓缓下降滑落的样子——这就是你看着一个智障向着智能演化的神明视角心跳脉搏！

准备好迎接明天第三天我们要开始学习的**大杀器精巧微型整容大法——LoRA 特化**，以及把模型变成高德地图极度省时省力还能提速的超级并行等工程技法了吗！我们将越战越勇！
