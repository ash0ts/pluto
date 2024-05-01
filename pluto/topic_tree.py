from pydantic import BaseModel, Field
import litellm
import json
import uuid
from typing import List, Optional
from .utils import extract_list
from .prompts import TREE_GENERATION_PROMPT
from .posthog.events import capture_event
import weave

class TopicTreeArguments(BaseModel):
    root_prompt: str
    model_system_prompt: str
    tree_degree: int = Field(default=10, gt=0)
    tree_depth: int = Field(default=3, gt=0)

class TopicTree(weave.Model):
    args: TopicTreeArguments = None
    tree_paths: List[List[str]] = []

    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, args: TopicTreeArguments):
        super().__init__()
        self.args = args
        self.tree_paths = []

    @weave.op()
    def build_tree(self, model_name: str = "gpt-3.5-turbo-0125"):
        build_id = uuid.uuid4()
        capture_event("build-tree", dict(model=model_name, tree_degree=self.args.tree_degree, tree_depth=self.args.tree_depth, build_id=build_id))
        self.tree_paths = self.build_subtree(model_name, [self.args.root_prompt], self.args.model_system_prompt, self.args.tree_degree, self.args.tree_depth)
        capture_event("build-tree-finished", dict(build_id=build_id))
        return self.tree_paths

    @weave.op()
    def build_subtree(self, model_name: str, node_path: List[str], system_prompt: str, tree_degree: int, subtree_depth: int) -> List[List[str]]:
        print(f"building subtree for path: {' -> '.join(node_path)}")
        if subtree_depth == 0:
            return [node_path]
        else:
            subnodes = self.get_subtopics(system_prompt=system_prompt, node_path=node_path, num_subtopics=tree_degree, model_name=model_name)
            updated_node_paths = [node_path + [sub] for sub in subnodes]
            result = []
            for path in updated_node_paths:
                result.extend(self.build_subtree(model_name, path, system_prompt, tree_degree, subtree_depth-1))
            return result

    @weave.op()
    def get_subtopics(self, system_prompt: str, node_path: List[str], num_subtopics: int, model_name: str) -> List[str]:
        prompt = TREE_GENERATION_PROMPT

        prompt = prompt.replace("{{{{system_prompt}}}}", system_prompt)
        prompt = prompt.replace("{{{{subtopics_list}}}}", ' -> '.join(node_path))
        prompt = prompt.replace("{{{{num_subtopics}}}}", str(num_subtopics))

        response = litellm.completion(
            model=model_name,
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return extract_list(response.choices[0].message.content)



    @weave.op()
    def save(self, save_path: str):
        path_dicts = []
        with open(save_path, "w") as f:
            for path in self.tree_paths:
                path_dict = dict(path=path)
                path_dicts.append(path_dict)
                f.write(json.dumps(path_dict)+"\n")
        dataset = weave.Dataset(name=save_path, rows=path_dicts)
        weave.publish(dataset)
        return dataset

