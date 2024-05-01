from typing import List, Dict, ClassVar
import json
from .utils import remove_linebreaks_and_spaces
from pydantic import BaseModel
import weave

class Dataset(BaseModel):
    samples: List[Dict] = []

    class Config:
        arbitrary_types_allowed = True
        extra = "ignore"

    def __init__(self):
        super().__init__()
        self.samples = []

    @classmethod
    def from_jsonl(cls, file_path):
        instance = cls()
        with open(file_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                assert cls.validate_sample(sample)
                instance.samples.append(sample)

        return instance

    @classmethod
    def from_list(cls, sample_list: List[Dict]):
        instance = cls()
        for sample in sample_list:
            assert self.validate_sample(sample)
            instance.samples.append(sample)

        return instance

    @classmethod
    def validate_sample(cls, sample: Dict):
        if 'messages' not in sample:
            return False
        for message in sample['messages']:
            if 'role' not in message or 'content' not in message:
                return False
            if message['role'] not in ['user', 'assistant', 'system']:
                return False
        return True

    # @weave.op()
    def save(self, save_path: str):
        samples = []
        with open(save_path, "w") as f:
            for sample in self.samples:
                cleaned_sample = remove_linebreaks_and_spaces(json.dumps(sample))
                samples.append(json.loads(cleaned_sample))
                f.write(cleaned_sample+"\n")
        dataset = weave.Dataset(name=save_path, rows=samples)
        weave.publish(dataset)
        
        print(f"saved dataset to {save_path}. You can now upload and fine-tune models on multiple platforms:\n\nHaven: https://app.haven.run/\nOpenAI: https://platform.openai.com/finetune")
        return dataset

    # @weave.op()
    def add_samples(self, samples: List[Dict]):
        sample_additions = []
        for sample in samples:
            if self.validate_sample(sample):
                sample_additions.append(sample)
            else:
                print("Invalid sample, not added:", sample)
        self.samples.extend(sample_additions)
