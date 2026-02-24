🔍 The Real Difference (This Is Important)

You are comparing:

Kwai-Kolors Kolors

Zhipu AI ChatGLM-6B

The key difference is:

Kolors does NOT use vanilla HuggingFace ChatGLM.

Kolors includes a modified ChatGLM implementation, often:

Custom modeling file

Custom attention

Hardcoded fp16 in some layers

Built assuming CUDA execution

While chatglm-6b from HuggingFace:

Is clean Transformers implementation

Fully CPU compatible in float32


