\# BMEN 600 Project



\## Team 11



\## Team Members Present

Pahan

Amir

Lex

Mahyar



\## Candidate Project

\### Biomedical Problem



For people with complex mobility needs, decoding EEG signals can be one of the only ways to help give them increased mobility and independence. This can often be done by decoding imagined movement and converting those signals to actions, such as moving a powered wheelchair. However, current motor imagery decoders struggle with accuracy. Further, the motor imagery decoder deployments are often power hungry, thus leading to difficulties deploying them at the edge.



\### Possible Research Question



How do RSSNs perform compared to current EEG motor imagery classifiers on accuracy and power efficiency in regards to motor imagery inference?



\### Possible Datasets



Physionet MI

5 class @ 160Hz x 64 channels for 3s trials

https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html



BCI Compeititon IV 2a

4 class @ 250Hz x 25 channels for 4s trials

https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014\_001.html



\### Biggest Uncertainty



The computing power to train and evaluate the model could be a big problem. This could slow down our research, and especially with the short timeline of this project it could be an issue. Further, because RSNNs have never been applied to EEG BCI data, there is no certainty that it doesn't work. 



\### Background Research

* motorSRNN: A spiking recurrent neural network inspired by brain topology for the effective and efficient decoding of cortical spike trains
* Surrogate Gradient Learning in Spiking Neural Networks
* https://github.com/fmi-basel/neural-decoding-RSNN
* https://arxiv.org/pdf/2409.01762

