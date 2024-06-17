from abc import ABC, abstractmethod
from warnings import warn
import torch
from datasets import load_dataset
from .retrieval import DataStore

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except Exception as e:
            warn(
                f"Loading the dataset failed with {str(e)}. This task will use a locally downloaded dataset, not from the HF hub. \
                This is expected behavior for the DS-1000 benchmark but not for other benchmarks!"
            )

        self.dstore = None

    def create_datastore(self, embedding_model_checkpoint="facebook/dragon-plus-query-encoder",
                         index_path="/home/rsadhukh/indexes/vault/flatindexIP.index", 
                         dataset_dir_or_name="Fsoft-AIC/the-vault-function:train_full-python",
                         text_col="docstring",
                         cont_col="code", 
                         enc_pool_strategy="cls", 
                         device=torch.device("cuda:0")):
        self.dstore = DataStore(embedding_model_checkpoint, index_path, 
                                dataset_dir_or_name, text_col, cont_col,
                                enc_pool_strategy=enc_pool_strategy, 
                                device=device)

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_prompt_with_fewshots(self, doc, num_shots):
        """Builds the prompt for the LM to generate from with few-shot examples.
        :param doc: dict[str: str]
            sample from the test dataset
        :param num_shots: int
            number of few-shot examples to include
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]
