from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification
from transformers import Pipeline
import torch
from transformers import pipeline

class MyPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]

        postprocess_kwargs = {}
        if "top_k" in kwargs:
            postprocess_kwargs["top_k"] = kwargs["top_k"]
        return preprocess_kwargs, {}, postprocess_kwargs

    def preprocess(self, inputs, args=2):
        model_input = torch.Tensor(inputs["input_ids"])
        return {"model_input": model_input}
    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs
    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class
    
PIPELINE_REGISTRY.register_pipeline(
    "new-task",
    pipeline_class=MyPipeline,
    pt_model=AutoModelForSequenceClassification,
    tf_model=TFAutoModelForSequenceClassification,
    type="text",
)


pipeline = pipeline("new-task", model='model_w/Qwen05', device=-1)
# returns 3 most likely labels
pipeline("This is the best meal I've ever had", top_k=3)