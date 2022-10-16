# 2D Primary Visual Cortex Mean-Field Model

Using Voltage-sensitive dye imaging (VSDI) [1, 2], Muller _et al._ showed that the stimulus-evoked population response in the primary visual cortex of the awake monkey performing visual tasks propagates as a travelling wave, with consistent dynamics across trials [3]. Chemla _et al._ further showed that these propagating waves are suppressive and suggested that their role is to increase the acuity of the visual system when presented with ambiguous stimuli [4].

This repository implements a theoretical model that accounts for the aforementioned phenomena. While a detailed model of a neocortical column has been published [5], its computational cost does not allow its generalization to a 1−2 cm² spatial scale.
The present model uses a mean-field formalism derived from networks of spiking neurons (adaptive exponential and fire AdEx model). Under this approach, proposed by El Boustani and Destexhe [6], the evolution of interconnected populations of neurons are described. Previous work in the lab allowed for the construction of 1-dimensional mean-field ring models [7, 8]. Indeed, the ring geometry offers a simple framework to investigate the emergence of spatio-temporal patterns of activity [9].
The present implementation extends these models: a network of two cortical excitatory and inhibitory neurons populations, connected with physiological connectivity profiles, is spatially arranged in a 2-dimensional geometry. The stimulus-evoked response of the model is computed under mean-field formalism.

---

## References

[1] A. Grinvald et al., "Vsdi: a new era in functional imaging of cortical dynamics", Nature Reviews Neuroscience, 2004

[2] S. Chemla et al., "Improving voltage-sensitive dye imaging: with a little help from computational approaches", Neurophotonics, 2017.

[3] L. Muller et al., "The stimulus-evoked population response in visual cortex of awake monkey is a propagating wave", Nature Communications, 2014.

[4] S. Chemla et al., "Suppressive traveling waves shape representations of illusory motion in primary visual cortex of awake primate", The Journal of Neuroscience, 2019.

[5] H. Markram et al., "Reconstruction and simulation of neocortical microcircuitry", Cell, 2015.

[6] E. Boustani and Destexhe, "A master equation formalism for macroscopic modeling of asynchronous irregular activity states", Neural Computation, 2009.

[7] Y. Zerlaut et al., "Modeling mesoscopic cortical dynamics using a mean-field model of conductance-based networks of adaptive exponential integrate-and-fire neuron", Journal of Computational Neuroscience, 2017.

[8] M. di Volo et al., "Biologically realistic mean-field models of conductance based networks of spiking neurons with adaptation", Neural Computation, 2019.

[9] Hansel and Sompolinsky, "Chaos and synchrony in a model of a hypercolumn in visual cortex", Journal of Computational Neuroscience, 1995.

---

## Requirements

- numpy

---

## Examples
