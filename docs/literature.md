# Literature table (starter, verifiable sources)

Curated starting bibliography. Only well-known, checkable sources are included —
**no invented citations**. Complete and extend while writing the report; each
entry records how it maps to this project.

| Ref | Author(s) / Year | Title / Venue | Why it is relevant here |
|---|---|---|---|
| [1] | Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. | FAO-56: Crop evapotranspiration. *FAO Irrigation and Drainage Paper 56*, 1998 | Reference ET0 (Penman–Monteith) used by the virtual-farm generator; soil-water balance framing |
| [2] | Doorenbos, J., & Pruitt, W. O. | Guidelines for predicting crop water requirements. *FAO Irrigation and Drainage Paper 24*, 1977 | Traditional crop-coefficient scheduling baseline (context for threshold policies) |
| [3] | Hochreiter, S., & Schmidhuber, J. | Long Short-Term Memory. *Neural Computation 9(8)*, 1997 | LSTM backbone for moisture-sequence forecasting |
| [4] | Simonyan, K., & Zisserman, A. | Very Deep Convolutional Networks for Large-Scale Image Recognition. *ICLR*, 2015 | Convolutional feature-stacking rationale behind the CNN-LSTM hybrid |
| [5] | Lim, B., & Zohren, S. | Time Series Forecasting with Deep Learning: A Survey. *Phil. Trans. R. Soc. A 379*, 2021 | Survey justifying deep-forecast baselines and hybrid architectures |
| [6] | Lakshminarayanan, B., Pritzel, A., & Blundell, C. | Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*, 2017 | Deep-ensemble uncertainty estimator (member std → intervals) |
| [7] | Kendall, A., & Gal, Y. | What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? *NeurIPS*, 2017 | Aleatoric/epistemic uncertainty discussion behind the confidence gates |
| [8] | Koenker, R., & Bassett, G. | Regression Quantiles. *JASA 73(361)*, 1978 | Pinball loss underpinning the quantile-CNN-LSTM intervals |
| [9] | Angelopoulos, A. N., & Bates, S. | A Gentle Introduction to Conformal Prediction... *arXiv:2107.07511*, 2021 | Distribution-free calibration ideas for coverage tuning (reported gap vs. target) |
| [10] | Liu, F. T., Ting, K. M., & Zhou, Z.-H. | Isolation Forest. *ICDM*, 2008 | Optional ML component of the sensor-fault detector |
| [11] | Zhou, Z.-H. | Ensemble Methods: Foundations and Algorithms. CRC Press, 2012 | General ensemble rationale (also horizon ensembling ideas) |
| [12] | Kamilaris, A., & Prenafeta-Boldú, F. X. | Deep learning in agriculture: A survey. *Computers and Electronics in Agriculture 147*, 2018 | Context for AI in agriculture; gaps this prototype addresses |

Usage map → report chapters: [1–2] data & physics; [3–5] forecasting;
[6–9] uncertainty & calibration; [10] fault detection; [11] evaluation robustness;
[12] positioning.