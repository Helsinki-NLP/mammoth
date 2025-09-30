# MAMMOTH Distributed Training Flow Chart

**Configuration**: 2 Nodes × 4 GPUs = 8 GPUs Total
**Tasks**: 8 Translation tasks (en→{ar,eu,fi,he,hi,hr,is,sw})
**Architecture**: Shared encoder + Language-specific decoders

## Node and Task Distribution

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MAMMOTH DISTRIBUTED TRAINING SETUP                        │
│                                 2 Nodes × 4 GPUs = 8 GPUs Total                    │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│            NODE 0               │    │            NODE 1               │
│    (Physical Machine 1)         │    │    (Physical Machine 2)         │
├─────────────────────────────────┤    ├─────────────────────────────────┤
│  GPU 0: task_en_ar (en→ar)      │    │  GPU 4: task_en_hi (en→hi)      │
│  GPU 1: task_en_eu (en→eu)      │    │  GPU 5: task_en_hr (en→hr)      │
│  GPU 2: task_en_fi (en→fi)      │    │  GPU 6: task_en_is (en→is)      │
│  GPU 3: task_en_he (en→he)      │    │  GPU 7: task_en_sw (en→sw)      │
└─────────────────────────────────┘    └─────────────────────────────────┘
           │                                        │
           └───────────── TCP Network ──────────────┘
                    (master_ip:master_port)
```

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    SHARED ENCODER ["shared"] - ALL-REDUCE                          │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐           │
│  │GPU 0│  │GPU 1│  │GPU 2│  │GPU 3│  │GPU 4│  │GPU 5│  │GPU 6│  │GPU 7│           │
│  │ ar  │  │ eu  │  │ fi  │  │ he  │  │ hi  │  │ hr  │  │ is  │  │ sw  │           │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘           │
│     │        │        │        │        │        │        │        │              │
│     └────────┼────────┼────────┼────────┼────────┼────────┼────────┘              │
│              └────────┼────────┼────────┼────────┼────────┘                       │
│                       └────────┼────────┼────────┘                                │
│                                └────────┘                                         │
│  Gradients: All-Reduce across all 8 GPUs → Same encoder weights everywhere        │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    LANGUAGE-SPECIFIC DECODERS - NO COMMUNICATION                   │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐           │
│  │ ar  │  │ eu  │  │ fi  │  │ he  │  │ hi  │  │ hr  │  │ is  │  │ sw  │           │
│  │Decd │  │Decd │  │Decd │  │Decd │  │Decd │  │Decd │  │Decd │  │Decd │           │
│  │GPU 0│  │GPU 1│  │GPU 2│  │GPU 3│  │GPU 4│  │GPU 5│  │GPU 6│  │GPU 7│           │
│  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘           │
│     ↑        ↑        ↑        ↑        ↑        ↑        ↑        ↑              │
│  Local    Local    Local    Local    Local    Local    Local    Local             │
│  Only     Only     Only     Only     Only     Only     Only     Only             │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Training Pipeline Flow

**Per GPU Process (×8):**

```
┌─────────────────────┐
│   Data Producer     │ ──┐
│   (Subprocess)      │   │ Queue(40) + Semaphore(40)
│ - Load en→X data    │   │
│ - Tokenize          │   │
│ - Apply transforms  │   │
└─────────────────────┘   ▼
                     ┌─────────────────────┐
                     │   Training Consumer │
                     │   (Main Process)    │
                     │ - Get batch from Q  │
                     │ - Forward pass      │
                     │ - Compute gradients │
                     │ - Sync shared comps │
                     │ - Update parameters │
                     └─────────────────────┘
```

## Communication Patterns Per Training Step

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        COMMUNICATION PATTERNS PER STEP                             │
└─────────────────────────────────────────────────────────────────────────────────────┘

Step 1: Task Sampling (Weighted)
├─ TaskQueueManager samples one task per GPU
├─ Example: GPU0→en_ar, GPU1→en_fi, GPU2→en_eu, etc.
└─ All-Gather task assignments across all 8 GPUs

Step 2: Forward Pass (Independent)
├─ Each GPU processes its batch locally
├─ Shared encoder: same weights, different inputs
└─ Language decoders: completely independent

Step 3: Gradient Computation (Local)
├─ Each GPU computes gradients for its components
├─ Encoder grads: computed by all 8 GPUs
└─ Decoder grads: computed only by owning GPU

Step 4: Gradient Synchronization (All-Reduce)
├─ Shared Encoder: All-Reduce across all 8 GPUs
│  └─ Sum gradients from all devices → average → broadcast
├─ Language Decoders: No communication (local only)
└─ Statistics: All-Gather training metrics

Step 5: Parameter Update (Local)
├─ Apply averaged encoder gradients on all GPUs
├─ Apply local decoder gradients on respective GPUs
└─ All GPUs have identical encoder, unique decoders
```

## Network Communication Details

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            NETWORK COMMUNICATION                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘

Inter-Node (Node 0 ↔ Node 1):
├─ TCP connection via master_ip:master_port
├─ All-Reduce operations for shared encoder
├─ All-Gather for statistics and task coordination
└─ Bandwidth: ~10MB buffer chunks (optimized)

Intra-Node (GPUs 0-3 on Node 0, GPUs 4-7 on Node 1):
├─ High-speed interconnect (NVLink/InfiniBand)
├─ Same All-Reduce/All-Gather operations
└─ Higher bandwidth, lower latency than inter-node
```

## Key Configuration Parameters

From `puhti/train.yaml`:

- **Queue Size**: 40 (per GPU)
- **World Size**: 8 (total GPUs)
- **Nodes**: 2
- **Batch Size**: 8 (per GPU)
- **Task Distribution**: `weighted_sampling`
- **Shared Components**: Encoder `["shared"]`
- **Language-Specific**: Decoders `["ar", "eu", "fi", "he", "hi", "hr", "is", "sw"]`

## Architecture Benefits

1. **Memory Efficiency**: Language-specific decoders mean each GPU only stores one decoder while sharing encoder computation
2. **Communication Optimization**: Only shared encoder requires expensive all-reduce across nodes
3. **Queue Management**: 8×40=320 batches in flight provide substantial buffering
4. **Fault Tolerance**: Individual task failures don't affect other language pairs
5. **Scalability**: Easy to add/remove language pairs without architectural changes

---
*Generated from MAMMOTH distributed training configuration analysis*