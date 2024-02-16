# -*- coding: utf-8 -*-
import logging.handlers
import os
from abc import ABC
from typing import List

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from ts.torch_handler.base_handler import BaseHandler
from pathlib import Path

PYTHON_LOGGER = logging.getLogger(__name__)
Path("log").mkdir(exist_ok=True)
HDLR = logging.handlers.TimedRotatingFileHandler("log/handler.log", when="midnight", backupCount=60)
STREAM_HDLR = logging.StreamHandler()
FORMATTER = logging.Formatter("%(asctime)s %(filename)s [%(levelname)s] %(message)s")
HDLR.setFormatter(FORMATTER)
STREAM_HDLR.setFormatter(FORMATTER)
PYTHON_LOGGER.addHandler(HDLR)
PYTHON_LOGGER.addHandler(STREAM_HDLR)
PYTHON_LOGGER.setLevel(logging.DEBUG)

# Absolute path to the folder location of this python file
FOLDER_ABSOLUTE_PATH = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))


class LangDetection(BaseHandler, ABC):
    """
    Handler for helsinki models
    """

    def __init__(self):
        super(LangDetection, self).__init__()
        self.initialized = False
        self.pipline = None
        self.manifest = None

    def initialize(self, context):
        """
        In this initialize function, the helsinki models and his tokenizer is loaded and
        Args:
            context: It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        PYTHON_LOGGER.info("Start loading xlm-roberta-base-language-detection")
        model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
        model = BetterTransformer.transform(model)
        tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
        model.eval()

        self.manifest = context.manifest
        properties = context.system_properties

        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.pipline = pipeline("text-classification", tokenizer=tokenizer, model=model, device=self.device)

        PYTHON_LOGGER.info("model load loaded successfully on the device {}".format(self.device))
        self.initialized = True

    def preprocess(self, requests):
        """
        Extract parameters from the request source lang, destination lang
        Tokenize all sentences.

        !!! Important note: One request equal on translation into one lang !!!

        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess

        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        sentences = []
        input_size_in_bytes = 0
        # Get all sentences to process
        for idx, data in enumerate(requests):
            try:
                inputs = data["body"]
                if isinstance(inputs, list):
                    sentences.extend(inputs)
                else:
                    sentences.append(inputs)
            except KeyError:
                input_text = data.get("data")
                if isinstance(input_text, (bytes, bytearray)):
                    input_text = input_text.decode("utf-8", errors="ignore")
                sentences.append(input_text)
                # remove multiple spaces

        for i in range(len(sentences)):
            sentences[i] = " ".join(sentences[i].split()).strip()
            input_size_in_bytes += len(sentences[i])
            self.context.metrics.add_size("NumberOfChars", input_size_in_bytes, None, "B")
        return sentences

    def inference(self, input_batch: List, *args, **kwargs) -> List:
        """
        Pass batch throw helsinki model
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
            *args: Not used
            **kwargs: Not used

        Returns: Translated sentences
        """
        with torch.no_grad():
            out = self.pipline(input_batch, max_length=128)
        return [out]

    def postprocess(self, inference_output):
        """
        Post Process Function
        """
        torch.cuda.empty_cache()
        return inference_output
