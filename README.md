# An End2End demand forecasting method based on transformer

Traditional sample-based interpretable models cannot handle the chronological order of features, and in most cases only have local interpretability on partial datasets. The attention-based approach is able to take into account both the chronological order of features and the ability to interpret globally when performing interpretable analysis. In addition, this end-to-end model I propose is able to adaptively change model complexity and can greatly improve the model reusability.

Why I chose the transformer method? Because transformers are very good at finding long-term dependencies and processing multi-modal information, which is exactly aligned with the nature of demand forecasting.

And there are more specific reasons like that I can do customization on the classic transformer to better fit the demand forecasting scenario. Let me show you three examples here.

One, I knew that demand forecasting has a variety of different user scenarios, some simple, some complex which led me to choose models of varying levels of complexity, choosing the right model and doing a test comparison can take a lot of effort. So, I added things like a gated residual network, which gave me the ability to automatically simplify the model. It can be powerful enough, and can also greatly simplify the complexity of the model when needed

Two, there are also many data types, internal, external, structured, or unstructured. Common forecasting methods can only process certain types of data, this prevents me from leveraging other useful data, which eventually affects prediction accuracy. So I added things like a variable selection network and temporal self-attention mechanism, which gave me the ability to individually process and fuse different multi-modal features, such as static, and time series data. 

THREE, demand forecasting is not only about demand and predicting results. It is necessary to conduct sufficient interpretability analysis of the predicted results to allow users to fully understand and use them with confidence.

## Getting started

To make it easy for you to get started with transformer tft, here's a list of recommended next steps.

Read these two papers first:

[Attention is all you need](https://arxiv.org/abs/1706.03762)

[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)

## Environment

Pipfile and Pipfile.lock stores the environment settings for this project. 

## Steps

$ `python hyper_opt.py`
$ `python train.py`

## Interpretable analysis

All experiments results will be stored at the folder lightning_logs, log analysis tools can be used to show the prediction results and the interpretable analysis.

## Step by step instructions based on a food manufactur example

Please go to `tutorials/pep.ipynb` for more details.



