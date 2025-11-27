---
author: Faith O. Oyedemi
bibliography: references/refs.bib
csl: references/csl/cjp.csl
date: 2025-11-26
tags:
- parameter sharing
- artificial neural networks
- translational invariance
title: Parameter Sharing
---

{{\< katex \>}}

## A Travel Back in Time

Parameter sharing predates deep learning as we know it today. To build a
solid foundation for grasping this concept, let us go back in time to
the 1950s, when neurophysiologists David Hubel and Torsten Wiesel were
studying the cat's visual cortex \[1\]. They made several discoveries
that would go on to shape practices across many fields, from medicine to
machine learning.

- Feature Detectors: They observed that specific neurons respond to
  specific visual stimuli. Some neurons would only fire when they
  detected a line at a particular orientation, for example, horizontal
  versus vertical. If you are familiar with a Convolutional Neural
  Network (CNN), then you have already seen this principle in action.
- Columnar Organization: Neurons that responded to the same type of
  stimulus were grouped together, i.e., neurons were organized (by
  function) into columns within the visual cortex.
- Ocular Dominance Columns: Some neurons primarily received input from
  the left eye, while others were driven predominantly by the right eye.

Not only did their findings open new pathways for understanding and
treating visual disorders, but they also earned the researchers the 1981
Nobel Prize in Physiology or Medicine, shared with Roger W. Sperry.

The next stop on this timeline is 1969, when Kunihiko Fukushima \[2\]
proposed a multilayer visual feature detector inspired by the visual
systems of cats and monkeys. The chief design principle was that each
element of the network responded to a specific feature of the input
pattern, such as brightness contrast, a dot in the pattern, or a line
segment of a particular orientation.

More than two decades later (1988), Homma, Atlas, and Marks introduced
the use of convolution as a means of generalizing the formal function of
a neuron \[3\]. In the early 1990s, Yann LeCun's work on LeNet-5
demonstrated the effectiveness of CNNs \[3\] and parameter sharing for
image classification. The fact that these networks required
significantly fewer parameters than fully connected architectures made
the approach both practical and accessible for machine learning
applications.

Since then, various neural architectures have continued to exploit this
principle for its advantages, from recurrent neural networks (RNNs)
\[4\] to transformers \[5\] and multilayer-perceptron mixers (Mixer)
\[6\].

## Why Share Parameters?

Parameter sharing is such a fundamental strategy that entire research
studies have been devoted to examining its behaviour and implications
\[7--9\]. What, then, is parameter sharing?

One way to describe it is to say that "the same feature detector should
apply everywhere." In a CNN, this manifests as a kernel convolving
across an image while reusing the same weights. In RNNs, it appears as a
single set of parameters representing the same transformation applied
sequentially. In other words, CNNs share parameters spatially while RNNs
do so temporally.

In more recent architectures, input and output embeddings often share
parameters in transformers; for Mixers, token-mixing MLPs share
parameters across channels, while channel-mixing MLPs share parameters
across tokens (patches). Two primary gains arise from this principle:

- Data efficiency: A significantly reduced number of parameters implies
  that less data is required for effective learning.
- Useful inductive bias: Parameter sharing enforces translation, time,
  or permutation invariance, which often leads to improved
  generalization.

## Implementations of Parameter Sharing

Taxonomically, parameter sharing can be discussed in two lights. One is
structural parameter sharing and the other, in the light of multitask
learning. It is expedient at this point to clarify that in this article,
we are discussing structural parameter sharing so phrases such as *'hard
sharing'* or *'soft sharing'* will not be used. What we are discussing
is symmetry-enforced weight reuse inside a single task. If you prefer to
be formal, you can put it as

\$\$ f(x)*i = \\phi*{\\theta}(x) \\forall ; i, \$\$

where \$\\theta\$ represents the parameters applied at different
coordinates \$i\$. I hope that this clarifies any confusion. We will now
discuss some dominant forms. PyTorch will be used for demonstration.

### Spatial Sharing in the Convolutional Neural Network (CNN)

The core principle in CNN is to apply the same filter (kernel) across
different spatial locations of the input data (usually \\(X \\in
\\mathcal{R}\^{H \\times W \\times C} \\)). Unlike the case of a fully
connected network where each neuron in a layer is connected to every
input pixel, the CNN slides a small filter all over the input volume. In
this convolution process, a dot product produces a feature map. The
important thing to note here is that the same set of filter
(weights/parameters) is applied to every part of it. In other words,
that particular kernel is looking for the occurence of a specific
feature all over the image. It could be lines, dots, curves, colour etc.
Does this remind you of the visual cortex of a cat? It should. To
demonstrate this, we will

- Create some fictional grayscale image of shape (32,32,1);
- Create a fully connected layer to deal with it;
- Create a convolutional layer to deal with it;
- Compare the parameters.

#### Using a Fully Connected Layer

``` python
import torch.nn as nn

input_height = 32
input_width = 32
input_channels = 1

# Input size for the Fully Connected (FC) layer
fc_input_size = input_height * input_width * input_channels # 32 * 32 * 1 = 1024
fc_output_size = 1 # A single output neuron

fc_layer = nn.Linear(in_features=fc_input_size, out_features=fc_output_size)

# Calculate parameters for the Fully Connected layer
fc_params = sum(p.numel() for p in fc_layer.parameters() if p.requires_grad)

print(f"----- Fully Connected Layer -----")
print(f"Input Features: {fc_input_size}, Output Features: {fc_output_size}")
print(f"Total Parameters (weights + bias): {fc_params}") # 1024 neurons + 1 bias = 1025
print("-" * 40)
```

``` text
----- Fully Connected Layer -----
Input Features: 1024, Output Features: 1
Total Parameters (weights + bias): 1025
----------------------------------------
```

#### Using a Convolutional Layer (with parameter sharing)

``` python
import torch.nn as nn

input_height = 32
input_width = 32
input_channels = 1

# We use a 3x3 kernel and generate 1 output channel (1 filter)
conv_layer = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=3, stride=1, padding=0)

# Calculate parameters for the CNN layer
# The weights shape is [out_channels, in_channels, kernel_height, kernel_width]
# The number of parameters is the size of this weight tensor plus a bias term.
conv_params = sum(p.numel() for p in conv_layer.parameters() if p.requires_grad)

print(f"--- Convolutional Layer (3x3 kernel) ---")
print(f"Weight Shape: {conv_layer.weight.shape}") # [1, 1, 3, 3]
print(f"Total Parameters (weights + bias): {conv_params}") # (3*3) + 1 = 10
print("-" * 40)
```

``` text
--- Convolutional Layer (3x3 kernel) ---
Weight Shape: torch.Size([1, 1, 3, 3])
Total Parameters (weights + bias): 10
----------------------------------------
```

#### Convfirming that the Same Weights Are Used

``` python
import torch
import torch.nn as nn

# Create a simple Conv2d layer
conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=False)

# Manually set the weights to a specific value for easy tracking (e.g., all ones)
# The shape is [out_channels, in_channels, kernel_height, kernel_width] -> [1, 1, 2, 2]
conv.weight.data = torch.ones_like(conv.weight.data)

# Define a simple 3x3 input image
# shape [batch_size, channels, height, width]
input_image = torch.tensor([[[
    [1., 2., 3.],
    [4., 5., 6.],
    [7., 8., 9.]
]]], dtype=torch.float32)

print(f"Input Image:\n{input_image[0, 0]}\n")
print(f"Kernel Weights (all ones):\n{conv.weight.data[0, 0]}\n")

# Apply the convolution
output_feature_map = conv(input_image)
```

If the same kernel is used all over the input, then we should expect the
following output:

  Quadrant      Output
  ------------- --------------------------------------------
  Upper left    (1 x 1) + (2 x 1) + (4 x 1) + (5 x 1) = 12
  Upper right   (2 x 1) + (3 x 1) + (5 x 1) + (6 x 1) = 16
  Lower left    (4 x 1) + (5 x 1) + (7 x 1) + (8 x 1) = 24
  Lower right   (5 x 1) + (6 x 1) + (8 x 1) + (9 x 1) = 28

``` python
print(f"Output Feature Map:\n{output_feature_map[0, 0]}\n")
```

``` text
Output Feature Map:
tensor([[12., 16.],
        [24., 28.]], grad_fn=<SelectBackward0>)
```

## Useful Bias: Neural Quantum States as a Case in Study

## Downsides to Parameter Sharing

------------------------------------------------------------------------

## References {#references .unnumbered}

:::::::::::: {#refs .references .csl-bib-body}
::: {#ref-HubelandWiesel .csl-entry}
[1 ]{.csl-left-margin}[HUBEL DH, WIESEL TN. [Brain and visual
perception: The story of a 25-year
collaboration](https://doi.org/10.1093/acprof:oso/9780195176186.001.0001).
Oxford University Press. 2004.]{.csl-right-inline}
:::

::: {#ref-Fukushima .csl-entry}
[2 ]{.csl-left-margin}[Fukushima K. 1969. [Visual feature extraction by
a multilayered network of analog threshold
elements](https://doi.org/10.1109/TSSC.1969.300225). IEEE Transactions
on Systems Science and Cybernetics. **5**(4): 322.]{.csl-right-inline}
:::

::: {#ref-Hommaetal .csl-entry}
[3 ]{.csl-left-margin}[Homma T, Atlas LE, Marks RJ. [An artificial
neural network for spatio-temporal bipolar patterns: Application to
phoneme
classification](https://proceedings.neurips.cc/paper_files/paper/1987/file/853f7b3615411c82a2ae439ab8c4c96e-Paper.pdf)
Neural information processing systems. vol 0, D Anderson, ed. American
Institute of Physics.]{.csl-right-inline}
:::

::: {#ref-Rumelhartetal .csl-entry}
[4 ]{.csl-left-margin}[Rumelhart DE, Hinton GE, Williams RJ. 1986.
[Learning representations by back-propagating
errors](https://doi.org/10.1038/323533a0). Nature.
**323**:]{.csl-right-inline}
:::

::: {#ref-Vaswanietal .csl-entry}
[5 ]{.csl-left-margin}[Vaswani A, Shazeer N, Parmar N, Uszkoreit J,
Jones L, Gomez AN, Kaiser Ł, Polosukhin I. [Attention is all you
need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
Advances in neural information processing systems. vol 30, I Guyon, U V
Luxburg, S Bengio, H Wallach, R Fergus, S Vishwanathan and R Garnett,
ed. Curran Associates, Inc.]{.csl-right-inline}
:::

::: {#ref-tolstikhin2021mlpmixerallmlparchitecturevision .csl-entry}
[6 ]{.csl-left-margin}[Tolstikhin I et al. 2021. [MLP-mixer: An all-MLP
architecture for
vision](https://arxiv.org/abs/2105.01601).]{.csl-right-inline}
:::

::: {#ref-ullrich2017softweightsharingneuralnetwork .csl-entry}
[7 ]{.csl-left-margin}[Ullrich K, Meeds E, Welling M. 2017. [Soft
weight-sharing for neural network
compression](https://arxiv.org/abs/1702.04008).]{.csl-right-inline}
:::

::: {#ref-Gaoetal .csl-entry}
[8 ]{.csl-left-margin}[Gao S, Deng C, Huang H. [Cross domain model
compression by structurally weight
sharing](https://doi.org/10.1109/CVPR.2019.00918) pp
8965--74.]{.csl-right-inline}
:::

::: {#ref-DabreandFujita .csl-entry}
[9 ]{.csl-left-margin}[Dabre R, Fujita A. 2021. [Recurrent stacking of
layers in neural networks: An application to neural machine
translation](https://doi.org/10.48550/arXiv.2106.10002).]{.csl-right-inline}
:::
::::::::::::
