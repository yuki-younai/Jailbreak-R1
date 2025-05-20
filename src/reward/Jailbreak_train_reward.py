import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional, Union
import torch.nn.functional as F
from src.reward.selfbule import SelfBleuReward
from src.reward.cosine_sim import CosineSentenceEmbeddingReward
from src.models.embedding import Embedding_API
from src.models.llm import LLM_local
import ast
import re



def standardize(lst):
    if len(lst)==0:
        return lst
    mean = sum(lst) / len(lst)
    variance = sum((x - mean) ** 2 for x in lst) / len(lst)
    std_dev = variance ** 0.5
    standardized_lst = [(x - mean) / (std_dev+1e-6) for x in lst]
    return standardized_lst

def assign_scores(lst):
    """
    Assign scores to elements in a list based on their values.
    The smallest element gets a score of 1, the largest gets a score of 0,
    and the rest are assigned scores in between based on their size.

    :param lst: List of numbers
    :return: A list of scores corresponding to the input list
    """
    sorted_lst = sorted(lst)
    n = len(lst)
    scores = [(i) / (n - 1) for i in range(n)]
    score_dict = dict(zip(sorted_lst, scores))
    return [score_dict[x] for x in lst]


class Jailbreak_Train_Reward(object):
    def __init__(
        self,
        model_name_or_path: str,
        device: Union[str, torch.device] = "cpu"
    ):
        super(Jailbreak_Train_Reward, self).__init__()
        self.device = device
        self.name_or_path = "warm_up_reward"
        self.__name__ = "warm_up_reward"
        self.condition = "<think> Sure, let's think step by step,"
        
        self.embedding_model = Embedding_API()
        self.bleu_model = SelfBleuReward()

    def to_device(self, device):
        self.device = device
        
    def get_diversity_scores(self, attack_list):
        self_bleu_score_list = []
        cos_sim_score_list = []
        if len(attack_list)>1:
            for idx_, select_attack in enumerate(attack_list):
                selfblue_model_attack = SelfBleuReward()
                selfblue_model_attack.append_reference(select_attack)
                new_arr = attack_list.copy()
                new_arr.pop(idx_)
                #self_bleu_score_1
                self_bleu_score_1 = selfblue_model_attack(new_arr)
                self_bleu_score_1 = sum(self_bleu_score_1)/len(self_bleu_score_1)
                #self_bleu_score_2
                selfblue_model_attack = SelfBleuReward()
                selfblue_model_attack.append_reference(new_arr)
                self_bleu_score_2 = selfblue_model_attack(select_attack)[0]
                self_bleu_score = (self_bleu_score_1+self_bleu_score_2)/2
                #cos_sim_score = self.embedding_model.compute_similarity([select_attack]*len(new_arr), new_arr)
                #cos_sim_score = sum(cos_sim_score)/len(cos_sim_score)
                cos_sim_score = 0
   
                self_bleu_score_list.append(1-self_bleu_score)
                cos_sim_score_list.append(1-cos_sim_score)
            self_bleu_score_list = standardize(self_bleu_score_list)
            cos_sim_score_list = standardize(cos_sim_score_list)
            div_scores_both = [a + b for a, b in zip(self_bleu_score_list, cos_sim_score_list)]
            div_scores_both = assign_scores(div_scores_both)
            div_scores_bleu = assign_scores(self_bleu_score_list)
            div_scores_sim = assign_scores(cos_sim_score_list)
        else:
            self_bleu_score_list = []
            cos_sim_score_list = []
            div_scores_both = [0.5]
            div_scores_bleu = [0.5]
            div_scores_sim = [0.5]
        
        return div_scores_both, div_scores_bleu, div_scores_sim
    
    def score(self, prompts, completions, attacks, thinks, consistency_scores, harmbench_scores, goal, **kwargs):
        
        for idx,score in enumerate(harmbench_scores):
            if score!=1:
                harmbench_scores[idx]=0
            
        select_idxs = []
        attack_list = []
        for idx, score in enumerate(consistency_scores):
            if score:
                select_idxs.append(idx)
                attack_list.append(attacks[idx])
        
        div_scores_both, div_scores_bleu, div_scores_sim = self.get_diversity_scores(attack_list)
        rewards_both = [0]*len(prompts)
        reward_bleu = [0]*len(prompts)
        reward_sim = [0]*len(prompts)
        for idx, select_idx in enumerate(select_idxs):
            rewards_both[select_idx] += div_scores_both[idx]+1
            reward_bleu[select_idx] += div_scores_bleu[idx]+1
            reward_sim[select_idx] += div_scores_sim[idx]+1

        rewards_both = [a + b for a, b in zip(rewards_both, harmbench_scores)]
        
        return rewards_both
